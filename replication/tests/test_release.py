"""Integrity tests for externally supplied archives and the dry-run boundary."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile

from replication.data import sha, unpack, verify

class DataArchiveTests(unittest.TestCase):
    def make_zip(self, folder, *, name="data/value.txt", payload=b"evidence\n",
                 digest=None, duplicate=False, symlink=False):
        target = folder/"bundle.zip"
        manifest = io.StringIO()
        writer = csv.writer(manifest, lineterminator="\n")
        writer.writerow(["path","bytes","sha256"])
        relative = name.removeprefix("data/")
        writer.writerow([relative,len(payload),digest or hashlib.sha256(payload).hexdigest()])
        with zipfile.ZipFile(target,"w") as archive:
            item = zipfile.ZipInfo(name)
            if symlink:
                item.create_system=3
                item.external_attr=(stat.S_IFLNK | 0o777)<<16
            archive.writestr(item,payload)
            if duplicate:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    archive.writestr(item,payload)
            archive.writestr("data/files_manifest.csv",manifest.getvalue())
        return target

    def test_valid_round_trip_and_nonempty_destination_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);archive=self.make_zip(base)
            with contextlib.redirect_stdout(io.StringIO()):
                final=unpack(archive,base/"install",sha(archive))
                report=verify(final)
            self.assertEqual((final/"value.txt").read_bytes(),b"evidence\n")
            self.assertEqual(report["files"],1)
            with self.assertRaises(FileExistsError):
                unpack(archive,base/"install",sha(archive))
            self.assertEqual((final/"value.txt").read_bytes(),b"evidence\n")

    def test_bad_archive_checksum_extracts_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);archive=self.make_zip(base)
            with self.assertRaises(ValueError):
                unpack(archive,base/"install","0"*64)
            self.assertFalse((base/"install").exists())

    def test_payload_digest_is_checked_even_when_zip_crc_is_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);archive=self.make_zip(base,digest="0"*64)
            with self.assertRaises(ValueError):
                unpack(archive,base/"install",sha(archive))
            self.assertFalse((base/"install/data").exists())

    def test_unsafe_paths_duplicates_and_symlinks_are_rejected(self):
        attacks=[{"name":"data/../../escape"}, {"name":"/tmp/escape"},
                 {"name":"data/C:/escape"}, {"name":"data\\escape"},
                 {"duplicate":True}, {"symlink":True}]
        for attack in attacks:
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as temp:
                base=Path(temp);archive=self.make_zip(base,**attack)
                with self.assertRaises(ValueError):
                    unpack(archive,base/"install",sha(archive))
                self.assertFalse((base/"install/data").exists())
                self.assertFalse((base/"escape").exists())

    def test_cli_dry_run_does_not_read_data_or_create_outputs(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp:
            missing=Path(temp)/"missing"
            output=Path(temp)/"output"
            p=subprocess.run([sys.executable,str(root/"replicate.py"),"run",
                              "--data",str(missing),"--output",str(output),"--dry-run"],
                             capture_output=True,text=True,cwd=temp)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertIn("RQ5/rq5_reproduce.py",p.stdout.replace("\\","/"))
            self.assertFalse(output.exists())
            self.assertFalse(missing.exists())

if __name__=="__main__":
    unittest.main()
