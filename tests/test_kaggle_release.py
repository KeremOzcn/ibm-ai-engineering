"""Check the public notebook input boundary without model downloads."""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from kaggle_release_support import unpack_regular_tar, verified_download  # noqa: E402


class KaggleInputTests(unittest.TestCase):
    def test_changed_download_cannot_replace_a_reviewed_input(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "dataset.json"
            target.write_bytes(b"existing")
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"changed")):
                with self.assertRaisesRegex(ValueError, "changed"):
                    verified_download("https://example.org/data", target, "0" * 64)
            self.assertEqual(target.read_bytes(), b"existing")
            self.assertFalse(target.with_suffix(".json.download").exists())

    def test_matching_download_is_published_and_reused_without_network(self):
        payload = b"reviewed bytes"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "data"
            with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)):
                verified_download("https://example.org/data", target, digest)
            with patch("urllib.request.urlopen", side_effect=AssertionError("network called")):
                self.assertEqual(
                    verified_download("https://example.org/data", target, digest), target
                )

    def test_tar_rejects_traversal_and_links_before_extracting_any_file(self):
        for name, kind in [("../escape", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                archive = root / "bad.tar"
                with tarfile.open(archive, "w") as bundle:
                    good = tarfile.TarInfo("good.txt")
                    good.size = 2
                    bundle.addfile(good, io.BytesIO(b"ok"))
                    bad = tarfile.TarInfo(name)
                    bad.type = kind
                    bundle.addfile(bad)
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    unpack_regular_tar(archive, root / "extracted")
                self.assertFalse((root / "extracted/good.txt").exists())


if __name__ == "__main__":
    unittest.main()
