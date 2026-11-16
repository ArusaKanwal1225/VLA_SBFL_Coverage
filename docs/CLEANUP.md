# Cleanup record

The evaluation checkout retains reusable scientific code, tests, frozen
protocols, final analyses/table tools, and the portable replication interface.
Inputs/results live in the separately released data archive and ignored
local data/reproduced directories.

Original inputs, old outputs, exploratory/pilot code and simulator/model runners
were preserved outside the working repository:

    /home/ubuntu/replication_backups/20261116-original/archived/

Original versions of edited files, pre-cleanup Git status, applied-path records
and validation logs are in the parent backup directory. Directory moves were
used where possible; image-layer directories required successful copies before
their original paths could be removed. No other project or model environment
was cleaned.

The final data ZIP resides on the host's separate /home filesystem:

    /home/pvalle/vla_replication_release_20261116/vla-sbfl-data-20261116.zip

An initial draft prepared before authenticated GitHub access is retained in that
directory's superseded folder. Upload only the canonical final ZIP and checksum.

GitHub source commit 44b2113bf9055f7aa0e7dce52d394c88030586b3 was fetched and
incorporated before finalization. The prepared cleanup branch is
replication-package-20261116. The original main branch and all Git history are
preserved; no remote push or forced history rewrite was performed.

Tracked data deletions reduce the current revision's content, but historical
blobs remain in old commits. Prefer a shallow clone of the final code branch.
Do not run a global Docker/container cleanup to reproduce this package.

To recover an archived item, restore only its specific relative path after
checking that the destination does not exist. Do not restore all historical
outputs over new reproduction outputs.
