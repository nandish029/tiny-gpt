import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.download_model import (
    compute_sha256,
    download_model,
    is_immutable_revision,
    is_valid_sha256,
    main as download_main,
    sanitize_error_message,
    validate_and_resolve_local_path,
    validate_remote_filename,
)


class TestArtifactDownload(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.valid_sha = "abcdef1234567890abcdef1234567890abcdef12"
        self.valid_hash = hashlib.sha256(b"test-artifact-payload").hexdigest()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_manifest(self, manifest_data):
        path = os.path.join(self.temp_dir, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        return path

    # -------------------------------------------------------------------------
    # A. IMMUTABLE REVISION VALIDATION TESTS
    # -------------------------------------------------------------------------

    def test_is_immutable_revision_logic(self):
        """is_immutable_revision must accept only full 40-char hex commit SHAs without whitespace."""
        # Valid 40-char SHAs
        self.assertTrue(is_immutable_revision("abcdef1234567890abcdef1234567890abcdef12"))
        self.assertTrue(is_immutable_revision("0123456789ABCDEF0123456789abcdef01234567"))

        # Rejection of leading and trailing whitespace
        self.assertFalse(is_immutable_revision(" " + self.valid_sha))
        self.assertFalse(is_immutable_revision(self.valid_sha + " "))
        self.assertFalse(is_immutable_revision("  " + self.valid_sha + "  "))
        self.assertFalse(is_immutable_revision("\t" + self.valid_sha))
        self.assertFalse(is_immutable_revision(self.valid_sha + "\n"))

        # Short SHAs
        self.assertFalse(is_immutable_revision("ef3b7f3"))
        self.assertFalse(is_immutable_revision("1234567"))
        self.assertFalse(is_immutable_revision("abcdef1234567890"))

        # Branches and tags
        self.assertFalse(is_immutable_revision("main"))
        self.assertFalse(is_immutable_revision("master"))
        self.assertFalse(is_immutable_revision("dev"))
        self.assertFalse(is_immutable_revision("release"))
        self.assertFalse(is_immutable_revision("v1.0"))
        self.assertFalse(is_immutable_revision("v1.0.0"))

        # Null, whitespace, empty, and malformed strings
        self.assertFalse(is_immutable_revision(None))
        self.assertFalse(is_immutable_revision(""))
        self.assertFalse(is_immutable_revision("   "))
        self.assertFalse(is_immutable_revision("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"))  # non-hex
        self.assertFalse(is_immutable_revision("abcdef1234567890abcdef1234567890abcdef123456"))  # too long

    def test_unconfigured_revision_refusal(self):
        """Downloader must refuse to run if revision is null or empty."""
        for empty_rev in [None, "", "   "]:
            manifest = {
                "gpt1": {
                    "repo_id": "test-org/tiny-gpt",
                    "revision": empty_rev,
                    "artifacts": {
                        "checkpoint": {
                            "remote_filename": "gpt1.pt",
                            "local_path": "checkpoints/gpt1/gpt1_baseline.pt",
                            "sha256": self.valid_hash,
                        }
                    },
                }
            }
            manifest_path = self._create_manifest(manifest)
            with self.assertRaises(ValueError) as cm:
                download_model("gpt1", manifest_path=manifest_path, project_root=self.temp_dir)
            self.assertIn("unconfigured Hub revision", str(cm.exception))

    def test_mutable_or_whitespace_revision_rejection(self):
        """Downloader must refuse short SHAs, branches, tags, and whitespace-padded SHAs."""
        invalid_revisions = [
            " " + self.valid_sha,
            self.valid_sha + " ",
            "ef3b7f3",
            "1234567",
            "dev",
            "release",
            "v1.0",
            "main",
            "master",
            "abcdef1234567890abcdef1234567890abcdef1z",  # invalid hex char 'z'
        ]
        for invalid_rev in invalid_revisions:
            manifest = {
                "gpt1": {
                    "repo_id": "test-org/tiny-gpt",
                    "revision": invalid_rev,
                    "artifacts": {
                        "checkpoint": {
                            "remote_filename": "gpt1.pt",
                            "local_path": "checkpoints/gpt1/gpt1_baseline.pt",
                            "sha256": self.valid_hash,
                        }
                    },
                }
            }
            manifest_path = self._create_manifest(manifest)
            with self.assertRaises(ValueError) as cm:
                download_model("gpt1", manifest_path=manifest_path, project_root=self.temp_dir)
            self.assertIn("requires an immutable full 40-character hexadecimal Git commit SHA", str(cm.exception))

    # -------------------------------------------------------------------------
    # B. CHECKSUM FORMAT AND PREFLIGHT VALIDATION TESTS
    # -------------------------------------------------------------------------

    def test_is_valid_sha256_logic(self):
        """is_valid_sha256 must accept exactly 64-character hex strings without whitespace."""
        self.assertTrue(is_valid_sha256("0" * 64))
        self.assertTrue(is_valid_sha256("abcdef0123456789" * 4))
        self.assertTrue(is_valid_sha256("ABCDEF0123456789" * 4))

        # Rejection of whitespace
        self.assertFalse(is_valid_sha256(" " + ("0" * 64)))
        self.assertFalse(is_valid_sha256(("0" * 64) + " "))

        # Invalid lengths
        self.assertFalse(is_valid_sha256("0" * 63))
        self.assertFalse(is_valid_sha256("0" * 65))
        self.assertFalse(is_valid_sha256("0" * 32))  # MD5 length
        self.assertFalse(is_valid_sha256("dummyhash"))

        # Non-hex characters
        self.assertFalse(is_valid_sha256(("0" * 63) + "g"))
        self.assertFalse(is_valid_sha256(None))
        self.assertFalse(is_valid_sha256(""))

    def test_missing_checksum_refusal(self):
        """Downloader must refuse to download if any artifact lacks a SHA-256 hash."""
        manifest = {
            "gpt1": {
                "repo_id": "test-org/tiny-gpt",
                "revision": self.valid_sha,
                "artifacts": {
                    "checkpoint": {
                        "remote_filename": "gpt1.pt",
                        "local_path": "checkpoints/gpt1/gpt1_baseline.pt",
                        "sha256": None,
                    }
                },
            }
        }
        manifest_path = self._create_manifest(manifest)
        with self.assertRaises(ValueError) as cm:
            download_model("gpt1", manifest_path=manifest_path, project_root=self.temp_dir)
        self.assertIn("unconfigured 'sha256'", str(cm.exception))

    @patch("huggingface_hub.hf_hub_download")
    def test_malformed_checksum_rejected_before_download(self, mock_hf_download):
        """Malformed SHA-256 in manifest must be rejected during preflight before any download starts."""
        malformed_hashes = [
            "dummyhash",
            "d41d8cd98f00b204e9800998ecf8427e",  # MD5 (32 chars)
            " " + self.valid_hash,
            self.valid_hash + " ",
            self.valid_hash[:-1],  # 63 chars
            self.valid_hash + "0",  # 65 chars
            self.valid_hash[:-1] + "z",  # invalid hex char
        ]
        for bad_hash in malformed_hashes:
            manifest = {
                "gpt1": {
                    "repo_id": "test-org/tiny-gpt",
                    "revision": self.valid_sha,
                    "artifacts": {
                        "checkpoint": {
                            "remote_filename": "gpt1.pt",
                            "local_path": "checkpoints/gpt1/gpt1_baseline.pt",
                            "sha256": bad_hash,
                        }
                    },
                }
            }
            manifest_path = self._create_manifest(manifest)
            with self.assertRaises(ValueError) as cm:
                download_model("gpt1", manifest_path=manifest_path, project_root=self.temp_dir)
            self.assertIn("Checksum must be exactly 64 hexadecimal characters", str(cm.exception))
            mock_hf_download.assert_not_called()

    # -------------------------------------------------------------------------
    # C. PATH CONFINEMENT, SYMLINKS, AND TRAVERSAL PREVENTION TESTS
    # -------------------------------------------------------------------------

    def test_validate_and_resolve_local_path_legitimate(self):
        """Legitimate repository relative paths are resolved safely inside root."""
        root = self.temp_dir
        resolved = validate_and_resolve_local_path("checkpoints/gpt1/gpt1_baseline.pt", root)
        expected = str((Path(root).resolve() / "checkpoints" / "gpt1" / "gpt1_baseline.pt").resolve(strict=False))
        self.assertEqual(resolved, expected)

        resolved_tok = validate_and_resolve_local_path("data/processed/tokenizer.json", root)
        expected_tok = str((Path(root).resolve() / "data" / "processed" / "tokenizer.json").resolve(strict=False))
        self.assertEqual(resolved_tok, expected_tok)

    def test_local_path_parent_traversal_rejected(self):
        """Parent directory traversal ('..') in local_path is strictly rejected."""
        root = self.temp_dir
        traversal_paths = [
            "../../outside.pt",
            "checkpoints/../../outside.pt",
            "checkpoints/gpt1/../../../etc/passwd",
            "..\\outside.pt",
            "checkpoints\\..\\..\\outside.pt",
        ]
        for path in traversal_paths:
            with self.assertRaises(ValueError) as cm:
                validate_and_resolve_local_path(path, root)
            self.assertIn("parent traversal", str(cm.exception).lower())

    def test_local_path_absolute_rejected(self):
        """Absolute local destination paths are strictly rejected."""
        root = self.temp_dir
        absolute_paths = [
            "/etc/shadow",
            "/tmp/model.pt",
            "\\Windows\\System32\\bad.dll",
            "C:\\bad.pt",
            "D:\\outside\\model.pt",
        ]
        for path in absolute_paths:
            with self.assertRaises(ValueError) as cm:
                validate_and_resolve_local_path(path, root)
            self.assertTrue(
                "absolute paths are not permitted" in str(cm.exception)
                or "drive paths are not permitted" in str(cm.exception)
            )

    def test_symlink_resolving_outside_root_rejected(self):
        """A symlink pointing outside the project root must be rejected during path containment check."""
        outside_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(outside_dir, ignore_errors=True))

        symlink_dir = os.path.join(self.temp_dir, "symlink_dest")
        try:
            os.symlink(outside_dir, symlink_dir, target_is_directory=True)
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"Symlinks not supported or permitted on this platform/environment: {e}")

        malicious_path = os.path.join("symlink_dest", "model.pt")
        with self.assertRaises(ValueError) as cm:
            validate_and_resolve_local_path(malicious_path, self.temp_dir)
        self.assertIn("resolves outside project root", str(cm.exception).lower())

    def test_remote_filename_traversal_rejected(self):
        """Remote filenames with directory traversal or absolute paths are rejected."""
        unsafe_remote = [
            "../secret.pt",
            "../../etc/passwd",
            "..\\evil.pt",
            "/absolute/path.pt",
            "\\windows\\bad.pt",
            "C:\\bad.pt",
        ]
        for fn in unsafe_remote:
            with self.assertRaises(ValueError):
                validate_remote_filename(fn)

        # Safe remote filenames must pass
        self.assertEqual(validate_remote_filename("gpt1_baseline.pt"), "gpt1_baseline.pt")
        self.assertEqual(validate_remote_filename("checkpoints/gpt1_baseline.pt"), "checkpoints/gpt1_baseline.pt")

    @patch("huggingface_hub.hf_hub_download")
    def test_traversal_manifest_rejected_before_download(self, mock_hf_download):
        """Manifest with path traversal must fail during preflight before any download occurs."""
        manifest = {
            "gpt1": {
                "repo_id": "test-org/tiny-gpt",
                "revision": self.valid_sha,
                "artifacts": {
                    "checkpoint": {
                        "remote_filename": "gpt1.pt",
                        "local_path": "../../escaped.pt",
                        "sha256": self.valid_hash,
                    }
                },
            }
        }
        manifest_path = self._create_manifest(manifest)
        with self.assertRaises(ValueError) as cm:
            download_model("gpt1", manifest_path=manifest_path, project_root=self.temp_dir)
        self.assertIn("parent traversal", str(cm.exception).lower())
        mock_hf_download.assert_not_called()

        escaped_file = os.path.abspath(os.path.join(self.temp_dir, "..", "escaped.pt"))
        self.assertFalse(os.path.exists(escaped_file))

    # -------------------------------------------------------------------------
    # D. ERROR SANITIZATION TESTS
    # -------------------------------------------------------------------------

    def test_credential_masking(self):
        """Tokens and credentials must be masked from error strings."""
        raw_error = "401 Client Error: Invalid token hf_abcdefghijklmnopqrstuvwxyz1234567890 for url"
        masked = sanitize_error_message(raw_error)
        self.assertNotIn("hf_abcdefghijklmnopqrstuvwxyz1234567890", masked)
        self.assertIn("hf_***REDACTED***", masked)

    def test_main_masks_token_in_handled_exception(self):
        """Exceptions handled by main() must never leak tokens into stderr."""
        secret_token = "hf_SUPERSECRETTOKEN1234567890"

        with patch("sys.argv", ["download_model.py", "--model", "gpt1", "--manifest", "nonexistent.json"]):
            with patch("scripts.download_model.download_model") as mock_dl:
                mock_dl.side_effect = ValueError(f"Authorization error with credential {secret_token}")

                stderr_capture = io.StringIO()
                with patch("sys.stderr", stderr_capture):
                    with self.assertRaises(SystemExit) as cm:
                        download_main()
                    self.assertEqual(cm.exception.code, 1)

                output = stderr_capture.getvalue()
                self.assertNotIn(secret_token, output)
                self.assertIn("hf_***REDACTED***", output)

    # -------------------------------------------------------------------------
    # E. CORE FUNCTIONALITY & VERIFICATION TESTS
    # -------------------------------------------------------------------------

    def test_unconfigured_repo_refusal(self):
        """Downloader must refuse to run if repo_id is null or empty."""
        manifest = {
            "gpt1": {
                "repo_id": None,
                "revision": self.valid_sha,
                "artifacts": {
                    "checkpoint": {
                        "remote_filename": "gpt1.pt",
                        "local_path": "checkpoints/gpt1/gpt1_baseline.pt",
                        "sha256": self.valid_hash,
                    }
                },
            }
        }
        manifest_path = self._create_manifest(manifest)
        with self.assertRaises(ValueError) as cm:
            download_model("gpt1", manifest_path=manifest_path, project_root=self.temp_dir)
        self.assertIn("unconfigured repository ID", str(cm.exception))

    def test_missing_remote_filename_refusal(self):
        """Downloader must refuse to download if remote_filename is unconfigured."""
        manifest = {
            "gpt1": {
                "repo_id": "test-org/tiny-gpt",
                "revision": self.valid_sha,
                "artifacts": {
                    "checkpoint": {
                        "remote_filename": None,
                        "local_path": "checkpoints/gpt1/gpt1_baseline.pt",
                        "sha256": self.valid_hash,
                    }
                },
            }
        }
        manifest_path = self._create_manifest(manifest)
        with self.assertRaises(ValueError) as cm:
            download_model("gpt1", manifest_path=manifest_path, project_root=self.temp_dir)
        self.assertIn("unconfigured 'remote_filename'", str(cm.exception))

    @patch("huggingface_hub.hf_hub_download")
    def test_successful_sha256_verification(self, mock_hf_download):
        """Valid download matching expected SHA-256 is successfully placed at target location."""
        content = b"canonical trained model parameters"
        content_hash = hashlib.sha256(content).hexdigest()

        def fake_download(repo_id, filename, revision, local_dir):
            file_path = os.path.join(local_dir, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(content)
            return file_path

        mock_hf_download.side_effect = fake_download

        manifest = {
            "gpt1": {
                "repo_id": "test-org/tiny-gpt",
                "revision": self.valid_sha,
                "artifacts": {
                    "checkpoint": {
                        "remote_filename": "gpt1_baseline.pt",
                        "local_path": "checkpoints/gpt1/gpt1_baseline.pt",
                        "sha256": content_hash,
                    }
                },
            }
        }
        manifest_path = self._create_manifest(manifest)
        result = download_model("gpt1", manifest_path=manifest_path, project_root=self.temp_dir)

        self.assertTrue(result)
        target_path = os.path.join(self.temp_dir, "checkpoints", "gpt1", "gpt1_baseline.pt")
        self.assertTrue(os.path.isfile(target_path))
        with open(target_path, "rb") as f:
            self.assertEqual(f.read(), content)
        mock_hf_download.assert_called_once()

    @patch("huggingface_hub.hf_hub_download")
    def test_hash_mismatch_rejection(self, mock_hf_download):
        """If downloaded content does not match manifest SHA-256, it must fail and not place the file."""
        content = b"tampered or corrupted content"

        def fake_download(repo_id, filename, revision, local_dir):
            file_path = os.path.join(local_dir, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(content)
            return file_path

        mock_hf_download.side_effect = fake_download

        manifest = {
            "gpt1": {
                "repo_id": "test-org/tiny-gpt",
                "revision": self.valid_sha,
                "artifacts": {
                    "checkpoint": {
                        "remote_filename": "gpt1_baseline.pt",
                        "local_path": "checkpoints/gpt1/gpt1_baseline.pt",
                        "sha256": "0" * 64,
                    }
                },
            }
        }
        manifest_path = self._create_manifest(manifest)

        with self.assertRaises(ValueError) as cm:
            download_model("gpt1", manifest_path=manifest_path, project_root=self.temp_dir)

        self.assertIn("SHA-256 verification failed", str(cm.exception))
        target_path = os.path.join(self.temp_dir, "checkpoints", "gpt1", "gpt1_baseline.pt")
        self.assertFalse(os.path.exists(target_path))

    @patch("huggingface_hub.hf_hub_download")
    def test_reuse_already_verified_local_file(self, mock_hf_download):
        """If local file exists and matches expected SHA-256, download is skipped entirely."""
        content = b"local verified checkpoint"
        content_hash = hashlib.sha256(content).hexdigest()

        target_path = os.path.join(self.temp_dir, "checkpoints", "gpt1", "gpt1_baseline.pt")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(content)

        manifest = {
            "gpt1": {
                "repo_id": "test-org/tiny-gpt",
                "revision": self.valid_sha,
                "artifacts": {
                    "checkpoint": {
                        "remote_filename": "gpt1_baseline.pt",
                        "local_path": "checkpoints/gpt1/gpt1_baseline.pt",
                        "sha256": content_hash,
                    }
                },
            }
        }
        manifest_path = self._create_manifest(manifest)
        result = download_model("gpt1", manifest_path=manifest_path, project_root=self.temp_dir)

        self.assertTrue(result)
        mock_hf_download.assert_not_called()

    @patch("huggingface_hub.hf_hub_download")
    def test_download_failure_handling(self, mock_hf_download):
        """Download network or Hub error raises a descriptive RuntimeError."""
        mock_hf_download.side_effect = ConnectionError("Could not reach Hugging Face servers")

        manifest = {
            "gpt1": {
                "repo_id": "test-org/tiny-gpt",
                "revision": self.valid_sha,
                "artifacts": {
                    "checkpoint": {
                        "remote_filename": "gpt1_baseline.pt",
                        "local_path": "checkpoints/gpt1/gpt1_baseline.pt",
                        "sha256": self.valid_hash,
                    }
                },
            }
        }
        manifest_path = self._create_manifest(manifest)

        with self.assertRaises(RuntimeError) as cm:
            download_model("gpt1", manifest_path=manifest_path, project_root=self.temp_dir)

        self.assertIn("Failed to download", str(cm.exception))
        self.assertIn("Could not reach Hugging Face servers", str(cm.exception))

    @patch("huggingface_hub.hf_hub_download")
    def test_prevention_of_unsafe_overwrite(self, mock_hf_download):
        """Corrupt download does NOT overwrite a preexisting local file."""
        original_content = b"preexisting valid artifact"
        target_path = os.path.join(self.temp_dir, "checkpoints", "gpt1", "gpt1_baseline.pt")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(original_content)

        corrupt_content = b"corrupted bytes from interrupted download"

        def fake_download(repo_id, filename, revision, local_dir):
            file_path = os.path.join(local_dir, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(corrupt_content)
            return file_path

        mock_hf_download.side_effect = fake_download

        manifest = {
            "gpt1": {
                "repo_id": "test-org/tiny-gpt",
                "revision": self.valid_sha,
                "artifacts": {
                    "checkpoint": {
                        "remote_filename": "gpt1_baseline.pt",
                        "local_path": "checkpoints/gpt1/gpt1_baseline.pt",
                        "sha256": "f" * 64,
                    }
                },
            }
        }
        manifest_path = self._create_manifest(manifest)

        with self.assertRaises(ValueError):
            download_model("gpt1", manifest_path=manifest_path, project_root=self.temp_dir)

        # Preexisting file must remain intact
        with open(target_path, "rb") as f:
            self.assertEqual(f.read(), original_content)

    def test_independent_configuration_all_three_models(self):
        """The canonical repository manifest independently configures gpt1, gpt2, and gpt3."""
        canonical_manifest_path = os.path.join(self.project_root, "configs", "artifact_manifest.json")
        self.assertTrue(os.path.isfile(canonical_manifest_path))

        with open(canonical_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for model in ["gpt1", "gpt2", "gpt3"]:
            self.assertIn(model, manifest)
            entry = manifest[model]
            self.assertIsNone(entry["repo_id"])
            self.assertIsNone(entry["revision"])

            artifacts = entry["artifacts"]
            for art_key in ["checkpoint", "tokenizer", "config"]:
                self.assertIn(art_key, artifacts)
                art = artifacts[art_key]
                self.assertIsNone(art["remote_filename"])
                self.assertIsNone(art["sha256"])
                self.assertIsNotNone(art["local_path"])

        # Checkpoints must point to distinct paths
        self.assertEqual(manifest["gpt1"]["artifacts"]["checkpoint"]["local_path"], "checkpoints/gpt1/gpt1_baseline.pt")
        self.assertEqual(manifest["gpt2"]["artifacts"]["checkpoint"]["local_path"], "checkpoints/gpt2/gpt2_baseline.pt")
        self.assertEqual(manifest["gpt3"]["artifacts"]["checkpoint"]["local_path"], "checkpoints/gpt3/gpt3_baseline.pt")

        # GPT-3 tokenizer must be distinct subword tokenizer
        self.assertEqual(manifest["gpt1"]["artifacts"]["tokenizer"]["local_path"], "data/processed/tokenizer.json")
        self.assertEqual(manifest["gpt2"]["artifacts"]["tokenizer"]["local_path"], "data/processed/tokenizer.json")
        self.assertEqual(manifest["gpt3"]["artifacts"]["tokenizer"]["local_path"], "data/processed/gpt3_tokenizer.json")

    def test_cli_unconfigured_refusal(self):
        """CLI refuses execution with returncode 1 when canonical manifest is unconfigured."""
        cmd = [sys.executable, "scripts/download_model.py", "--model", "gpt1"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unconfigured repository ID", result.stderr)


if __name__ == "__main__":
    unittest.main()
