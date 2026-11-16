# Publishing the data on Zenodo

Upload the canonical vla-sbfl-data-20261116.zip and its accompanying .sha256 file.
replication/archive.json records its exact size, checksum, source commit,
file count and verification status. Do not upload the superseded draft.

Suggested title: **VoxelSBFL: data for the paper replication package**.

Describe four VLA models, 96,000 Pick-Up episodes, 2,016 Move-Near candidates,
the final 1,109 target executions, 34 SBFL techniques, and CPU-only analysis.
State that the release includes frozen reconstructed Pick-Up maps for RQ5,
all final exact statistical reviews, and per-fold voxel-size diagnostics.
It contains no execution videos, checkpoints or simulator dependencies.

Use the paper's actual authors and the authors' chosen data license.
CITATION.cff lists the manuscript authors; no publication DOI or license is invented.

After publication:

1. Set url and doi in replication/archive.json to the actual Zenodo record values.
2. Link the final GitHub code commit/release from the Zenodo related identifiers.
3. Push the prepared code branch and merge/release it when ready.
4. Test download, unpack and tables from a shallow clean clone.

~~~bash
python replication/data.py download
python replication/data.py unpack vla-sbfl-data-20261116.zip
python replicate.py tables
~~~

Updated data require a new version and checksum. Do not replace a published
archive silently. The code's checksum is available before upload; the URL
is intentionally unset until publication.

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
