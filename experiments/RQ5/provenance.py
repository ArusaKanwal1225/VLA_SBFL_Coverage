"""Separate historical construction, frozen artifact integrity and compatibility.

The archived source text reproduces the historical full-code signature. Current
scientific modules are deliberately pinned byte-for-byte; selected RQ1 training
helpers are pinned by AST to allow unrelated reporting/statistical changes.
Changing this contract requires an explicit scientific review, not a hash refresh.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = Path(__file__).with_name('compatibility_contract.json')
HISTORICAL_SIGNATURE = '910e24f9b9d91fdd8863ed3353c12dd082a81de824292f140fc39e5435eeda71'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def ast_fingerprint(node):
    """Canonical AST independent of Python's display of empty optional fields."""
    def normalize(value):
        if isinstance(value, ast.AST):
            return {'type': type(value).__name__, **{
                key: normalize(item) for key, item in ast.iter_fields(value)
                if item is not None and item != []}}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, bytes):
            return {'bytes': value.hex()}
        if value is Ellipsis:
            return {'ellipsis': True}
        return value
    return digest(json.dumps(normalize(node), sort_keys=True, separators=(',', ':')).encode())


def verify_compatibility(root=ROOT):
    contract = json.loads(CONTRACT.read_text())
    record = json.loads((root/'experiments/RQ1/results/phase_a_primary/run.json').read_text())
    details = record['signature_details']
    for field, source in contract['historical_sources'].items():
        if digest(source.encode()) != details[field]:
            raise ValueError('Historical source fingerprint mismatch: '+field)
    signature = digest(json.dumps(details, sort_keys=True, separators=(',', ':')).encode())
    if (record['status'] != 'complete' or signature != HISTORICAL_SIGNATURE
            or record['experiment_signature'] != signature
            or contract['historical_signature'] != signature):
        raise ValueError('Historical signature mismatch')
    for relative, expected in contract['scientific_files_sha256'].items():
        if digest((root/relative).read_bytes()) != expected:
            raise ValueError('Scientific compatibility mismatch: '+relative)
    nodes = {}
    for node in ast.parse((root/'experiments/RQ1/run_experiment.py').read_text()).body:
        name = getattr(node, 'name', None)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in contract['runner_scientific_ast_sha256']:
            nodes[name] = ast_fingerprint(node)
    if nodes != contract['runner_scientific_ast_sha256']:
        raise ValueError('RQ1 training/helper compatibility mismatch')
    # Compare the runtime scientific parameters; code-only reporting hashes differ.
    from experiments.RQ1.run_experiment import experiment_signature
    from vla_sbfl.voxelizer import VoxelGrid
    _, current = experiment_signature(VoxelGrid(.02, (-1., -1., 0.), 'robot_base'),
                                      root/'experiments/RQ1/RQ1.md')
    ignored = {'runner_sha256', 'analysis_sha256', 'protocol_sha256'}
    if {k:v for k,v in current.items() if k not in ignored} != {k:v for k,v in details.items() if k not in ignored}:
        raise ValueError('Scientific parameter mismatch')
    return signature


def verify_frozen_maps(maps, root=ROOT):
    contract = json.loads(CONTRACT.read_text())
    manifest = root/'experiments/RQ5/data/files_manifest.csv'
    if digest(manifest.read_bytes()) != contract['artifact_manifest_sha256']:
        raise ValueError('Published artifact manifest mismatch')
    with manifest.open(newline='') as stream:
        entries = list(csv.DictReader(stream))
    prefix = 'reconstructed_pickup_maps/'
    count = 0
    expected_paths = set()
    for row in entries:
        if not row['path'].startswith(prefix):
            continue
        relative = Path(row['path'][len(prefix):])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe artifact path')
        expected_paths.add(relative.as_posix())
        path = Path(maps)/relative
        if path.stat().st_size != int(row['bytes']) or digest(path.read_bytes()) != row['sha256']:
            raise ValueError('Frozen map integrity mismatch: '+str(relative))
        count += path.name == 'metadata.json'
    actual = {p.relative_to(maps).as_posix() for p in Path(maps).rglob('*') if p.is_file()}
    if count != 320 or actual != expected_paths:
        raise ValueError('Incomplete or unlisted frozen map artifacts')
    return count
