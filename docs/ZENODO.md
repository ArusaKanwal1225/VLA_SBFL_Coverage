# Published data on Zenodo

- Record: https://zenodo.org/records/24280744
- Version DOI: https://doi.org/10.5281/zenodo.24280744
- File: [vla-sbfl-data.zip](https://zenodo.org/records/24280744/files/vla-sbfl-data.zip?download=1)
- Dataset license: CC BY 4.0, as stated in the Zenodo record.
- ZIP size: 3,426,674,917 bytes.
- SHA-256: `329453c7fb0dcd19934d0b3e3e44e0f5a073c88e12d0a98b0e57613a29343da0`.
- Zenodo MD5: `89c2ddf9a96a05ca55402eaeac279b60`.

The published file was renamed from the prepared `vla-sbfl-data-20261116.zip`
to `vla-sbfl-data.zip`. Its size and MD5 match the prepared archive; its content
and trusted SHA-256 are unchanged. `replication/archive.json` pins the published
version's direct download URL, DOI, size, checksum, and source provenance.
The dataset DOI identifies the data, not the paper or the replication software.

## Download and reproduce

From the repository root, after installing the analysis environment:

~~~bash
python replication/data.py download
python replication/data.py unpack vla-sbfl-data.zip
python replicate.py tables
~~~

The download verifies SHA-256 before accepting the ZIP. Extraction verifies
every payload against the manifest and refuses a nonempty existing data directory.
If downloading manually, pass the ZIP's local path to the unpack command.
The original dated filename also works if its contents have the trusted checksum.

The data cover four VLA models, 96,000 Pick-Up episodes, 2,016 Move-Near candidates,
and the final 1,109 selected target executions. They include the frozen reconstructed
Pick-Up maps used for RQ5, final exact statistical reviews, and per-fold voxel-size
diagnostics. Execution videos, checkpoints, and simulator dependencies are excluded.
See the root README and RQ5 provenance documentation for details.

## Future releases

Updated data require a new Zenodo version and checksum. Do not replace a published
archive silently. Update the record URL, direct file URL, DOI, filename, size, and
checksums in `replication/archive.json`, then update documentation and citations.
Link the corresponding GitHub code release from the Zenodo related identifiers.
Test downloading, unpacking, and table generation from a shallow clean clone.

To rebuild an archive from installed data:

~~~bash
python replication/package_data.py --data data --output /path/to/new-data.zip
~~~

The tool preserves payload bytes, regenerates the global manifest, and refuses
an existing output file. Source-data modifications require review and a new release.
The original RQ5 inner manifest and compatibility contract must not be refreshed
to disguise a changed frozen map.

The cleanup preserves Git history; old data blobs remain in earlier commits.
A remote history rewrite would require a separate coordinated action.
