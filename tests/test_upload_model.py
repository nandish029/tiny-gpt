import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.upload_model import (
    CANONICAL_HASHES,
    DEFAULT_REPO_ID,
    FILE_ALLOWLIST,
    FROZEN_CHECKPOINT_HASHES,
    compute_sha256,
    is_immutable_revision,
    is_valid_sha256,
    main as upload_main,
    prepare_manifest_data,
    sanitize_error_message,
    upload_artifacts,
    validate_allowlist,
    verify_local_artifacts,
)


class TestUploadModel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir).resolve()
        self.valid_commit_sha = "abcdef0123456789abcdef0123456789abcdef01"

        # Populate a minimal valid tree matching the 8 allowlisted files
        for rel_path, canonical_hash in CANONICAL_HASHES.items():
            dest = self.project_root / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Create a file whose SHA256 matches canonical_hash by patching compute_sha256 in tests,
            # or write dummy data and compute hash when needed
            with open(dest, "wb") as f:
                f.write(f"payload-for-{rel_path}".encode("utf-8"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. ALLOWLIST ENFORCEMENT TESTS
    # -------------------------------------------------------------------------

    def test_allowlist_exact_files(self):
        """Allowlist must contain exactly the 8 specified files."""
        self.assertEqual(len(FILE_ALLOWLIST), 8)
        expected_files = {
            "checkpoints/gpt1/gpt1_baseline.pt",
            "checkpoints/gpt2/gpt2_baseline.pt",
            "checkpoints/gpt3/gpt3_baseline.pt",
            "data/processed/tokenizer.json",
            "data/processed/gpt3_tokenizer.json",
            "configs/gpt1.yaml",
            "configs/gpt2.yaml",
            "configs/gpt3.yaml",
        }
        self.assertEqual(set(FILE_ALLOWLIST.keys()), expected_files)

    def test_validate_allowlist_accepts_valid_files(self):
        """validate_allowlist accepts any subset or full set of allowlisted files."""
        validate_allowlist(list(FILE_ALLOWLIST.keys()))
        validate_allowlist(["checkpoints/gpt1/gpt1_baseline.pt"])

    def test_validate_allowlist_rejects_unapproved_files(self):
        """validate_allowlist rejects unapproved files, parent traversal, or arbitrary paths."""
        unapproved = [
            "src/model/gpt.py",
            "scripts/train_gpt1.py",
            ".env",
            "checkpoints/gpt1/other.pt",
            "experiments/gpt1/results.md",
            "../secret.txt",
            "/etc/passwd",
        ]
        for bad_file in unapproved:
            with self.assertRaises(ValueError) as cm:
                validate_allowlist([bad_file])
            self.assertIn("not in the explicit 8-file allowlist", str(cm.exception))

    # -------------------------------------------------------------------------
    # 2. LOCAL FILE VERIFICATION & PATH INTEGRITY
    # -------------------------------------------------------------------------

    def test_verify_local_artifacts_missing_file(self):
        """verify_local_artifacts raises FileNotFoundError if any allowlisted file is missing."""
        # Remove one file
        missing_file = self.project_root / "checkpoints/gpt2/gpt2_baseline.pt"
        missing_file.unlink()

        with self.assertRaises(FileNotFoundError) as cm:
            verify_local_artifacts(project_root=str(self.project_root))
        self.assertIn("Required artifact not found", str(cm.exception))
        self.assertIn("gpt2_baseline.pt", str(cm.exception))

    def test_verify_local_artifacts_checkpoint_hash_mismatch(self):
        """verify_local_artifacts raises ValueError if a checkpoint hash differs from frozen baseline."""
        # Mock compute_sha256 to return non-matching hash for gpt1 checkpoint
        def mock_compute(path, chunk_size=65536):
            if "gpt1_baseline.pt" in path:
                return "0000000000000000000000000000000000000000000000000000000000000000"
            for rel, h in CANONICAL_HASHES.items():
                if rel in path.replace("\\", "/"):
                    return h
            return "1111111111111111111111111111111111111111111111111111111111111111"

        with patch("scripts.upload_model.compute_sha256", side_effect=mock_compute):
            with self.assertRaises(ValueError) as cm:
                verify_local_artifacts(project_root=str(self.project_root))
            self.assertIn("FROZEN CHECKPOINT MISMATCH", str(cm.exception))
            self.assertIn("gpt1_baseline.pt", str(cm.exception))

    def test_verify_local_artifacts_canonical_hash_mismatch(self):
        """verify_local_artifacts raises ValueError if a tokenizer or config differs from canonical hash."""
        def mock_compute(path, chunk_size=65536):
            if "tokenizer.json" in path:
                return "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
            for rel, h in CANONICAL_HASHES.items():
                if rel in path.replace("\\", "/"):
                    return h
            return "1111111111111111111111111111111111111111111111111111111111111111"

        with patch("scripts.upload_model.compute_sha256", side_effect=mock_compute):
            with self.assertRaises(ValueError) as cm:
                verify_local_artifacts(project_root=str(self.project_root))
            self.assertIn("CANONICAL ARTIFACT HASH MISMATCH", str(cm.exception))
            self.assertIn("tokenizer.json", str(cm.exception))

    def test_verify_local_artifacts_success(self):
        """verify_local_artifacts succeeds when all files exist and hashes match."""
        def mock_compute(path, chunk_size=65536):
            for rel, h in CANONICAL_HASHES.items():
                if rel in path.replace("\\", "/"):
                    return h
            raise ValueError(f"Unexpected path: {path}")

        with patch("scripts.upload_model.compute_sha256", side_effect=mock_compute):
            hashes = verify_local_artifacts(project_root=str(self.project_root))
            self.assertEqual(len(hashes), 8)
            for rel, h in CANONICAL_HASHES.items():
                self.assertEqual(hashes[rel], h)

    def test_local_artifacts_never_modified_or_renamed(self):
        """Preflight checks must never alter or rename local artifact files."""
        # Record mtimes and contents before
        original_mtimes = {}
        original_contents = {}
        for rel_path in FILE_ALLOWLIST:
            p = self.project_root / rel_path
            original_mtimes[rel_path] = p.stat().st_mtime
            with open(p, "rb") as f:
                original_contents[rel_path] = f.read()

        def mock_compute(path, chunk_size=65536):
            for rel, h in CANONICAL_HASHES.items():
                if rel in path.replace("\\", "/"):
                    return h
            raise ValueError(f"Unexpected path: {path}")

        with patch("scripts.upload_model.compute_sha256", side_effect=mock_compute):
            verify_local_artifacts(project_root=str(self.project_root))

        # Verify untouched after
        for rel_path in FILE_ALLOWLIST:
            p = self.project_root / rel_path
            self.assertTrue(p.exists(), f"File {rel_path} was removed or renamed!")
            self.assertEqual(p.stat().st_mtime, original_mtimes[rel_path])
            with open(p, "rb") as f:
                self.assertEqual(f.read(), original_contents[rel_path])

    # -------------------------------------------------------------------------
    # 3. UPLOAD EXECUTION & MOCK INTERACTIONS
    # -------------------------------------------------------------------------

    def test_dry_run_mode_makes_no_hub_calls(self):
        """Dry-run mode verifies files but never invokes Hub API methods."""
        mock_api = MagicMock()

        def mock_compute(path, chunk_size=65536):
            for rel, h in CANONICAL_HASHES.items():
                if rel in path.replace("\\", "/"):
                    return h
            raise ValueError(f"Unexpected path: {path}")

        with patch("scripts.upload_model.compute_sha256", side_effect=mock_compute):
            result = upload_artifacts(
                repo_id=DEFAULT_REPO_ID,
                project_root=str(self.project_root),
                dry_run=True,
                api=mock_api,
            )

        self.assertTrue(result["dry_run"])
        self.assertEqual(result["files_verified"], 8)
        mock_api.create_commit.assert_not_called()
        mock_api.upload_file.assert_not_called()
        mock_api.list_repo_files.assert_not_called()

    def test_successful_mocked_upload(self):
        """Successful upload creates atomic commit, retrieves commit SHA, and verifies remote presence."""
        mock_api = MagicMock()
        mock_commit_info = MagicMock()
        mock_commit_info.oid = self.valid_commit_sha
        mock_api.create_commit.return_value = mock_commit_info

        # Remote repo files list contains all 8 remote paths
        mock_api.list_repo_files.return_value = list(FILE_ALLOWLIST.values()) + [".gitattributes", "README.md"]

        def mock_compute(path, chunk_size=65536):
            for rel, h in CANONICAL_HASHES.items():
                if rel in path.replace("\\", "/"):
                    return h
            raise ValueError(f"Unexpected path: {path}")

        with patch("scripts.upload_model.compute_sha256", side_effect=mock_compute):
            result = upload_artifacts(
                repo_id=DEFAULT_REPO_ID,
                project_root=str(self.project_root),
                commit_message="Release Tiny GPT models",
                dry_run=False,
                api=mock_api,
            )

        self.assertFalse(result["dry_run"])
        self.assertEqual(result["commit_sha"], self.valid_commit_sha)
        self.assertEqual(result["repo_id"], DEFAULT_REPO_ID)

        # Verify create_commit was called with all 8 operations
        mock_api.create_commit.assert_called_once()
        _, kwargs = mock_api.create_commit.call_args
        self.assertEqual(kwargs["repo_id"], DEFAULT_REPO_ID)
        self.assertEqual(kwargs["commit_message"], "Release Tiny GPT models")
        operations = kwargs["operations"]
        self.assertEqual(len(operations), 8)
        remote_destinations = [op.path_in_repo for op in operations]
        self.assertEqual(set(remote_destinations), set(FILE_ALLOWLIST.values()))

        # Verify list_repo_files was called with the exact commit SHA
        mock_api.list_repo_files.assert_called_once_with(
            repo_id=DEFAULT_REPO_ID, revision=self.valid_commit_sha
        )

    def test_upload_failure_error_handling(self):
        """Upload network or server failure raises RuntimeError."""
        mock_api = MagicMock()
        mock_api.create_commit.side_effect = Exception("500 Internal Server Error: upload endpoint timed out")

        def mock_compute(path, chunk_size=65536):
            for rel, h in CANONICAL_HASHES.items():
                if rel in path.replace("\\", "/"):
                    return h
            raise ValueError(f"Unexpected path: {path}")

        with patch("scripts.upload_model.compute_sha256", side_effect=mock_compute):
            with self.assertRaises(RuntimeError) as cm:
                upload_artifacts(
                    repo_id=DEFAULT_REPO_ID,
                    project_root=str(self.project_root),
                    dry_run=False,
                    api=mock_api,
                )
            self.assertIn("Hugging Face Hub upload failed", str(cm.exception))
            self.assertIn("500 Internal Server Error", str(cm.exception))

    def test_remote_presence_verification_missing_file(self):
        """If remote repository listing is missing any uploaded file, abort with RuntimeError."""
        mock_api = MagicMock()
        mock_commit_info = MagicMock()
        mock_commit_info.oid = self.valid_commit_sha
        mock_api.create_commit.return_value = mock_commit_info

        # Omit one remote file from listing
        partial_remote_files = [p for p in FILE_ALLOWLIST.values() if "gpt3_baseline.pt" not in p]
        mock_api.list_repo_files.return_value = partial_remote_files

        def mock_compute(path, chunk_size=65536):
            for rel, h in CANONICAL_HASHES.items():
                if rel in path.replace("\\", "/"):
                    return h
            raise ValueError(f"Unexpected path: {path}")

        with patch("scripts.upload_model.compute_sha256", side_effect=mock_compute):
            with self.assertRaises(RuntimeError) as cm:
                upload_artifacts(
                    repo_id=DEFAULT_REPO_ID,
                    project_root=str(self.project_root),
                    dry_run=False,
                    api=mock_api,
                )
            self.assertIn("Remote verification failed", str(cm.exception))
            self.assertIn("gpt3_baseline.pt", str(cm.exception))

    def test_invalid_commit_sha_returned(self):
        """If Hub returns an invalid or mutable revision string, abort with ValueError."""
        mock_api = MagicMock()
        mock_commit_info = MagicMock()
        # Invalid commit SHA (branch name instead of 40-char SHA)
        mock_commit_info.oid = "main"
        mock_api.create_commit.return_value = mock_commit_info

        def mock_compute(path, chunk_size=65536):
            for rel, h in CANONICAL_HASHES.items():
                if rel in path.replace("\\", "/"):
                    return h
            raise ValueError(f"Unexpected path: {path}")

        with patch("scripts.upload_model.compute_sha256", side_effect=mock_compute):
            with self.assertRaises(ValueError) as cm:
                upload_artifacts(
                    repo_id=DEFAULT_REPO_ID,
                    project_root=str(self.project_root),
                    dry_run=False,
                    api=mock_api,
                )
            self.assertIn("malformed or non-commit revision", str(cm.exception))

    # -------------------------------------------------------------------------
    # 4. CREDENTIAL SANITIZATION TESTS
    # -------------------------------------------------------------------------

    def test_sanitize_error_message(self):
        """Tokens, api keys, and bearer tokens must be stripped from error messages."""
        raw_msg = (
            "API request failed with token=hf_9876543210abcdef9876543210 "
            "and Authorization: Bearer hf_secret_bearer_token and api-key=my_secret_key"
        )
        cleaned = sanitize_error_message(raw_msg)
        self.assertNotIn("hf_9876543210abcdef9876543210", cleaned)
        self.assertNotIn("hf_secret_bearer_token", cleaned)
        self.assertNotIn("my_secret_key", cleaned)
        self.assertIn("token=***REDACTED***", cleaned)
        self.assertIn("Bearer ***REDACTED***", cleaned)
        self.assertIn("api-key=***REDACTED***", cleaned)

    def test_upload_failure_masks_token_in_exception(self):
        """When HfApi raises an exception containing an auth token, it must be masked."""
        mock_api = MagicMock()
        mock_api.create_commit.side_effect = Exception(
            "Authentication failed for request with token=hf_SuperSecretToken123456789"
        )

        def mock_compute(path, chunk_size=65536):
            for rel, h in CANONICAL_HASHES.items():
                if rel in path.replace("\\", "/"):
                    return h
            raise ValueError(f"Unexpected path: {path}")

        with patch("scripts.upload_model.compute_sha256", side_effect=mock_compute):
            with self.assertRaises(RuntimeError) as cm:
                upload_artifacts(
                    repo_id=DEFAULT_REPO_ID,
                    project_root=str(self.project_root),
                    dry_run=False,
                    api=mock_api,
                )
            err = str(cm.exception)
            self.assertNotIn("hf_SuperSecretToken123456789", err)
            self.assertIn("token=***REDACTED***", err)

    def test_cli_masks_credentials_on_error(self):
        """CLI main entrypoint must mask credentials when printing error to stderr."""
        with patch(
            "scripts.upload_model.verify_local_artifacts",
            side_effect=RuntimeError("Token failed: hf_SecretValue9999999999"),
        ):
            captured_stderr = io.StringIO()
            with patch("sys.stderr", captured_stderr), patch("sys.argv", ["upload_model.py", "--dry-run"]):
                with self.assertRaises(SystemExit) as cm:
                    upload_main()
                self.assertEqual(cm.exception.code, 1)

            err_output = captured_stderr.getvalue()
            self.assertNotIn("hf_SecretValue9999999999", err_output)
            self.assertIn("hf_***REDACTED***", err_output)

    # -------------------------------------------------------------------------
    # 5. MANIFEST PREPARATION TESTS
    # -------------------------------------------------------------------------

    def test_prepare_manifest_structure(self):
        """prepare_manifest_data prepares exact remote filenames, hashes, and leaves revision unconfigured."""
        manifest = prepare_manifest_data(repo_id="binod029/tiny-gpt", revision=None)

        self.assertIn("gpt1", manifest)
        self.assertIn("gpt2", manifest)
        self.assertIn("gpt3", manifest)

        # Revision must be None (unconfigured until actual commit SHA exists)
        self.assertIsNone(manifest["gpt1"]["revision"])
        self.assertIsNone(manifest["gpt2"]["revision"])
        self.assertIsNone(manifest["gpt3"]["revision"])

        # Check GPT-1 artifacts
        gpt1_artifacts = manifest["gpt1"]["artifacts"]
        self.assertEqual(
            gpt1_artifacts["checkpoint"]["remote_filename"], "checkpoints/gpt1/gpt1_baseline.pt"
        )
        self.assertEqual(
            gpt1_artifacts["checkpoint"]["sha256"], FROZEN_CHECKPOINT_HASHES["checkpoints/gpt1/gpt1_baseline.pt"]
        )
        self.assertEqual(
            gpt1_artifacts["tokenizer"]["remote_filename"], "tokenizers/tokenizer.json"
        )
        self.assertEqual(
            gpt1_artifacts["tokenizer"]["sha256"], CANONICAL_HASHES["data/processed/tokenizer.json"]
        )
        self.assertEqual(
            gpt1_artifacts["config"]["remote_filename"], "configs/gpt1.yaml"
        )
        self.assertEqual(
            gpt1_artifacts["config"]["sha256"], CANONICAL_HASHES["configs/gpt1.yaml"]
        )

        # Check GPT-2 artifacts
        gpt2_artifacts = manifest["gpt2"]["artifacts"]
        self.assertEqual(
            gpt2_artifacts["checkpoint"]["remote_filename"], "checkpoints/gpt2/gpt2_baseline.pt"
        )
        self.assertEqual(
            gpt2_artifacts["checkpoint"]["sha256"], FROZEN_CHECKPOINT_HASHES["checkpoints/gpt2/gpt2_baseline.pt"]
        )

        # Check GPT-3 artifacts
        gpt3_artifacts = manifest["gpt3"]["artifacts"]
        self.assertEqual(
            gpt3_artifacts["checkpoint"]["remote_filename"], "checkpoints/gpt3/gpt3_baseline.pt"
        )
        self.assertEqual(
            gpt3_artifacts["checkpoint"]["sha256"], FROZEN_CHECKPOINT_HASHES["checkpoints/gpt3/gpt3_baseline.pt"]
        )
        self.assertEqual(
            gpt3_artifacts["tokenizer"]["remote_filename"], "tokenizers/gpt3_tokenizer.json"
        )
        self.assertEqual(
            gpt3_artifacts["tokenizer"]["sha256"], CANONICAL_HASHES["data/processed/gpt3_tokenizer.json"]
        )
        self.assertEqual(
            gpt3_artifacts["config"]["remote_filename"], "configs/gpt3.yaml"
        )

    def test_prepare_manifest_with_explicit_revision(self):
        """prepare_manifest_data accepts an immutable commit SHA when explicitly passed."""
        manifest = prepare_manifest_data(
            repo_id="binod029/tiny-gpt", revision=self.valid_commit_sha
        )
        self.assertEqual(manifest["gpt1"]["revision"], self.valid_commit_sha)

    def test_prepare_manifest_rejects_mutable_revision(self):
        """prepare_manifest_data rejects mutable branches or short revisions."""
        for bad_rev in ["main", "v1.0", "dev", "1234567", " " + self.valid_commit_sha]:
            with self.assertRaises(ValueError) as cm:
                prepare_manifest_data(repo_id="binod029/tiny-gpt", revision=bad_rev)
            self.assertIn("mutable or malformed revision", str(cm.exception))

    def test_cli_refuses_to_overwrite_authoritative_manifest(self):
        """CLI refuses to automatically overwrite configs/artifact_manifest.json."""
        authoritative = (Path.cwd() / "configs" / "artifact_manifest.json").resolve()
        captured_stderr = io.StringIO()

        with patch("sys.stderr", captured_stderr), patch(
            "sys.argv",
            ["upload_model.py", "--save-prepared-manifest", "configs/artifact_manifest.json"],
        ):
            with self.assertRaises(SystemExit) as cm:
                upload_main()
            self.assertEqual(cm.exception.code, 1)

        err_output = captured_stderr.getvalue()
        self.assertIn("Refusing to overwrite authoritative 'configs/artifact_manifest.json'", err_output)


if __name__ == "__main__":
    unittest.main()
