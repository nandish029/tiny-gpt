import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Default Hugging Face repository
DEFAULT_REPO_ID = "binod029/tiny-gpt"

# Strict 40-character hexadecimal commit SHA pattern
COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")

# Strict 64-character hexadecimal SHA-256 pattern
SHA256_HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")

# Explicit 8-file allowlist mapping: local relative path -> remote repo path
FILE_ALLOWLIST: dict[str, str] = {
    "checkpoints/gpt1/gpt1_baseline.pt": "checkpoints/gpt1/gpt1_baseline.pt",
    "checkpoints/gpt2/gpt2_baseline.pt": "checkpoints/gpt2/gpt2_baseline.pt",
    "checkpoints/gpt3/gpt3_baseline.pt": "checkpoints/gpt3/gpt3_baseline.pt",
    "data/processed/tokenizer.json": "tokenizers/tokenizer.json",
    "data/processed/gpt3_tokenizer.json": "tokenizers/gpt3_tokenizer.json",
    "configs/gpt1.yaml": "configs/gpt1.yaml",
    "configs/gpt2.yaml": "configs/gpt2.yaml",
    "configs/gpt3.yaml": "configs/gpt3.yaml",
}

# Frozen reference hashes for baseline checkpoints (must match bit-for-bit)
FROZEN_CHECKPOINT_HASHES: dict[str, str] = {
    "checkpoints/gpt1/gpt1_baseline.pt": "7e67e9774e57750ca8b1e7c47438fbe73ca9c55fb3c22ae68716dd030cb659ab",
    "checkpoints/gpt2/gpt2_baseline.pt": "984b85010f88a1f8e47a14acfb1a0d2e9e12625b9fa79175b8b5ea18899d0b48",
    "checkpoints/gpt3/gpt3_baseline.pt": "35d125f5936fd6fd72e0fd7a5fea2a72d44a34a5d88b17d1bb5c4870e54ba6ab",
}

# Authoritative canonical hashes for all 8 allowable files
CANONICAL_HASHES: dict[str, str] = {
    "checkpoints/gpt1/gpt1_baseline.pt": "7e67e9774e57750ca8b1e7c47438fbe73ca9c55fb3c22ae68716dd030cb659ab",
    "checkpoints/gpt2/gpt2_baseline.pt": "984b85010f88a1f8e47a14acfb1a0d2e9e12625b9fa79175b8b5ea18899d0b48",
    "checkpoints/gpt3/gpt3_baseline.pt": "35d125f5936fd6fd72e0fd7a5fea2a72d44a34a5d88b17d1bb5c4870e54ba6ab",
    "data/processed/tokenizer.json": "d5d37c711f1617804e39dcf3a7bd65444e5204c39239f8391316ca3d7b76d2e0",
    "data/processed/gpt3_tokenizer.json": "7d090effc428ab13cda17621f9ea253319a86e1800656744874652e440fcb793",
    "configs/gpt1.yaml": "ab68622722091925e3ba86461a27832f9fae0efa9f8e244ff26b22dea72ab8eb",
    "configs/gpt2.yaml": "ce5ef40ecd53fdb9dcf17a525165e5abac63fc0648e4f197c0a1ed5c1f2d919c",
    "configs/gpt3.yaml": "49c20cd47893a0ee596c2f3592813e146865a06eddff265ee754c0a6d5cd8ed8",
}


def sanitize_error_message(message: str) -> str:
    """Masks potential authentication tokens, keys, and credentials from error strings."""
    sanitized = re.sub(r"hf_[a-zA-Z0-9]{8,}", "hf_***REDACTED***", message)
    sanitized = re.sub(r"(token=)[^\s&]+", r"\1***REDACTED***", sanitized)
    sanitized = re.sub(r"(Bearer\s+)[^\s]+", r"\1***REDACTED***", sanitized)
    sanitized = re.sub(r"(api[_-]?key=)[^\s&]+", r"\1***REDACTED***", sanitized, flags=re.IGNORECASE)
    return sanitized


def compute_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Computes the SHA-256 hex digest of a local file in read-only mode."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def is_immutable_revision(revision: str | None) -> bool:
    """
    Returns True only if the revision is an explicit, immutable full 40-character
    hexadecimal Git commit SHA without leading or trailing whitespace.
    """
    if not revision or not isinstance(revision, str):
        return False
    return bool(COMMIT_SHA_PATTERN.fullmatch(revision))


def is_valid_sha256(hash_str: str | None) -> bool:
    """
    Returns True only if hash_str is an exact 64-character hexadecimal SHA-256 string
    without leading or trailing whitespace.
    """
    if not hash_str or not isinstance(hash_str, str):
        return False
    return bool(SHA256_HEX_PATTERN.fullmatch(hash_str))


def validate_allowlist(files: list[str]) -> None:
    """
    Validates that every file in the provided list is in the explicit 8-file allowlist.
    Rejects any unapproved files, parent directory traversal, or absolute paths.
    """
    for f in files:
        # Standardize path separators for validation
        normalized = f.replace("\\", "/")
        if normalized not in FILE_ALLOWLIST:
            raise ValueError(
                f"File '{f}' is not in the explicit 8-file allowlist. "
                f"Allowed files: {sorted(list(FILE_ALLOWLIST.keys()))}"
            )


def verify_local_artifacts(project_root: str | None = None) -> dict[str, str]:
    """
    Verifies that all 8 allowable files exist, are strictly confined within project_root,
    and match their canonical and frozen SHA-256 digests.

    Returns a dictionary mapping relative path to verified SHA-256 hex digest.
    Raises FileNotFoundError if any file is missing.
    Raises ValueError if any file violates allowlist containment or hash checks.
    """
    effective_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    computed_hashes: dict[str, str] = {}

    print("=== Preflight Verification of Local Artifacts ===")
    print(f"Project root: {effective_root}")
    print(f"Allowlist count: {len(FILE_ALLOWLIST)} files\n")

    # Phase 1: Validate path containment and verify all files exist
    resolved_paths: dict[str, Path] = {}
    for rel_path in FILE_ALLOWLIST:
        # Validate path safety
        if os.path.isabs(rel_path) or rel_path.startswith(("/", "\\")):
            raise ValueError(f"Unsafe local path '{rel_path}': absolute paths forbidden.")
        if re.match(r"^[a-zA-Z]:", rel_path):
            raise ValueError(f"Unsafe local path '{rel_path}': drive paths forbidden.")
        parts = re.split(r"[/\\]+", rel_path)
        if any(part == ".." for part in parts):
            raise ValueError(f"Unsafe local path '{rel_path}': traversal ('..') forbidden.")

        target_path = (effective_root / rel_path).resolve(strict=False)

        # Containment check
        try:
            target_path.relative_to(effective_root)
        except ValueError:
            raise ValueError(
                f"Local file '{rel_path}' resolves outside project root ({effective_root})."
            )

        if not target_path.is_file():
            raise FileNotFoundError(
                f"Required artifact not found: '{rel_path}' (resolved: {target_path})"
            )
        resolved_paths[rel_path] = target_path

    # Phase 2: Compute SHA-256 and verify against frozen & canonical hashes
    for rel_path, remote_path in FILE_ALLOWLIST.items():
        target_path = resolved_paths[rel_path]

        # Compute SHA-256 in read-only mode without modifying file
        file_hash = compute_sha256(str(target_path))
        computed_hashes[rel_path] = file_hash

        # Strict checkpoint baseline check
        if rel_path in FROZEN_CHECKPOINT_HASHES:
            expected_frozen = FROZEN_CHECKPOINT_HASHES[rel_path]
            if file_hash != expected_frozen:
                raise ValueError(
                    f"FROZEN CHECKPOINT MISMATCH for '{rel_path}':\n"
                    f"  Expected: {expected_frozen}\n"
                    f"  Found:    {file_hash}\n"
                    "Upload aborted. Checkpoint must match frozen reference hash."
                )

        # General canonical hash check
        if rel_path in CANONICAL_HASHES:
            expected_canonical = CANONICAL_HASHES[rel_path]
            if file_hash != expected_canonical:
                raise ValueError(
                    f"CANONICAL ARTIFACT HASH MISMATCH for '{rel_path}':\n"
                    f"  Expected: {expected_canonical}\n"
                    f"  Found:    {file_hash}\n"
                    "Upload aborted. Artifact differs from authoritative hash."
                )

        size_bytes = target_path.stat().st_size
        print(f"[OK] Verified: {rel_path} -> {remote_path} ({size_bytes:,} bytes, {file_hash[:8]}...)")

    print("\nAll 8 allowable files verified successfully against canonical hashes.\n")
    return computed_hashes


def prepare_manifest_data(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    computed_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Builds the manifest dictionary for gpt1, gpt2, and gpt3 according to the
    format in configs/artifact_manifest.json.

    Leaves revision as None unless a valid immutable commit SHA is provided.
    Does NOT invent revision or hashes.
    """
    hashes = computed_hashes if computed_hashes is not None else CANONICAL_HASHES

    # If revision is provided, ensure it is an immutable full 40-character commit SHA
    if revision is not None and not is_immutable_revision(revision):
        raise ValueError(
            f"Cannot configure manifest with mutable or malformed revision {revision!r}. "
            "Revision must be an immutable 40-character hexadecimal commit SHA or None."
        )

    return {
        "gpt1": {
            "repo_id": repo_id,
            "revision": revision,
            "artifacts": {
                "checkpoint": {
                    "remote_filename": FILE_ALLOWLIST["checkpoints/gpt1/gpt1_baseline.pt"],
                    "local_path": "checkpoints/gpt1/gpt1_baseline.pt",
                    "sha256": hashes.get("checkpoints/gpt1/gpt1_baseline.pt"),
                },
                "tokenizer": {
                    "remote_filename": FILE_ALLOWLIST["data/processed/tokenizer.json"],
                    "local_path": "data/processed/tokenizer.json",
                    "sha256": hashes.get("data/processed/tokenizer.json"),
                },
                "config": {
                    "remote_filename": FILE_ALLOWLIST["configs/gpt1.yaml"],
                    "local_path": "configs/gpt1.yaml",
                    "sha256": hashes.get("configs/gpt1.yaml"),
                },
            },
        },
        "gpt2": {
            "repo_id": repo_id,
            "revision": revision,
            "artifacts": {
                "checkpoint": {
                    "remote_filename": FILE_ALLOWLIST["checkpoints/gpt2/gpt2_baseline.pt"],
                    "local_path": "checkpoints/gpt2/gpt2_baseline.pt",
                    "sha256": hashes.get("checkpoints/gpt2/gpt2_baseline.pt"),
                },
                "tokenizer": {
                    "remote_filename": FILE_ALLOWLIST["data/processed/tokenizer.json"],
                    "local_path": "data/processed/tokenizer.json",
                    "sha256": hashes.get("data/processed/tokenizer.json"),
                },
                "config": {
                    "remote_filename": FILE_ALLOWLIST["configs/gpt2.yaml"],
                    "local_path": "configs/gpt2.yaml",
                    "sha256": hashes.get("configs/gpt2.yaml"),
                },
            },
        },
        "gpt3": {
            "repo_id": repo_id,
            "revision": revision,
            "artifacts": {
                "checkpoint": {
                    "remote_filename": FILE_ALLOWLIST["checkpoints/gpt3/gpt3_baseline.pt"],
                    "local_path": "checkpoints/gpt3/gpt3_baseline.pt",
                    "sha256": hashes.get("checkpoints/gpt3/gpt3_baseline.pt"),
                },
                "tokenizer": {
                    "remote_filename": FILE_ALLOWLIST["data/processed/gpt3_tokenizer.json"],
                    "local_path": "data/processed/gpt3_tokenizer.json",
                    "sha256": hashes.get("data/processed/gpt3_tokenizer.json"),
                },
                "config": {
                    "remote_filename": FILE_ALLOWLIST["configs/gpt3.yaml"],
                    "local_path": "configs/gpt3.yaml",
                    "sha256": hashes.get("configs/gpt3.yaml"),
                },
            },
        },
    }


def upload_artifacts(
    repo_id: str = DEFAULT_REPO_ID,
    project_root: str | None = None,
    commit_message: str | None = None,
    dry_run: bool = False,
    api: Any | None = None,
) -> dict[str, Any]:
    """
    Safely uploads the 8 allowlisted files to the Hugging Face repository.
    Strictly verifies all files and hashes before initiating upload.
    Retrieves the actual commit SHA and verifies remote file presence after upload.
    Keeps credentials masked and never modifies local files.
    """
    effective_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()

    if not repo_id or not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("Target repository ID must be a non-empty string.")
    clean_repo_id = repo_id.strip()

    # Preflight integrity checks: verify existence and hashes for all 8 files
    computed_hashes = verify_local_artifacts(str(effective_root))

    if dry_run:
        print("[DRY-RUN] Preflight validation passed for all 8 files. No network upload performed.")
        return {
            "dry_run": True,
            "repo_id": clean_repo_id,
            "files_verified": len(FILE_ALLOWLIST),
            "hashes": computed_hashes,
        }

    # Prepare operations for atomic commit
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required for uploading. Install with 'pip install huggingface_hub'."
        )

    client = api or HfApi()

    operations = [
        CommitOperationAdd(
            path_in_repo=remote_path,
            path_or_fileobj=str((effective_root / rel_path).resolve()),
        )
        for rel_path, remote_path in FILE_ALLOWLIST.items()
    ]

    msg = commit_message or "Upload canonical Tiny GPT models (GPT-1, GPT-2, GPT-3)"

    print(f"Uploading {len(operations)} files to Hugging Face repository '{clean_repo_id}'...")

    try:
        commit_info = client.create_commit(
            repo_id=clean_repo_id,
            operations=operations,
            commit_message=msg,
        )
    except Exception as e:
        cleaned_err = sanitize_error_message(str(e))
        raise RuntimeError(f"Hugging Face Hub upload failed: {cleaned_err}") from None

    # Retrieve and validate commit SHA
    commit_sha = getattr(commit_info, "oid", str(commit_info)).strip()
    if not is_immutable_revision(commit_sha):
        raise ValueError(
            f"Remote repository returned malformed or non-commit revision {commit_sha!r}. "
            "Expected full 40-character hexadecimal commit SHA."
        )

    print(f"[OK] Upload commit created: {commit_sha}")

    # Verify remote file presence
    print("Verifying remote file presence on Hugging Face Hub...")
    try:
        remote_files = client.list_repo_files(repo_id=clean_repo_id, revision=commit_sha)
    except Exception as e:
        cleaned_err = sanitize_error_message(str(e))
        raise RuntimeError(
            f"Failed to verify remote file presence at commit '{commit_sha}': {cleaned_err}"
        ) from None

    missing_remote = [path for path in FILE_ALLOWLIST.values() if path not in remote_files]
    if missing_remote:
        raise RuntimeError(
            f"Remote verification failed! The following uploaded files were not found "
            f"in repository '{clean_repo_id}' at commit '{commit_sha}': {missing_remote}"
        )

    print(f"[SUCCESS] All {len(FILE_ALLOWLIST)} files confirmed present at remote commit {commit_sha}.")

    return {
        "dry_run": False,
        "repo_id": clean_repo_id,
        "commit_sha": commit_sha,
        "remote_files": list(FILE_ALLOWLIST.values()),
        "hashes": computed_hashes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Safe Hugging Face Hub artifact uploader for Tiny GPT models."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Target Hugging Face repository ID (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform preflight verification of all 8 files without uploading.",
    )
    parser.add_argument(
        "--prepare-manifest",
        action="store_true",
        help="Print prepared manifest JSON with remote filenames and hashes to stdout.",
    )
    parser.add_argument(
        "--save-prepared-manifest",
        type=str,
        default=None,
        help="Optional path to write the prepared manifest JSON (refuses to overwrite configs/artifact_manifest.json).",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Commit message for Hugging Face Hub upload.",
    )
    parser.add_argument(
        "--execute-upload",
        action="store_true",
        help="Explicit flag required to initiate upload. Prevents accidental execution.",
    )

    args = parser.parse_args()

    # Guard against accidental execution: require either --dry-run, --prepare-manifest, or --execute-upload
    if not args.execute_upload and not args.dry_run and not args.prepare_manifest and not args.save_prepared_manifest:
        print(
            "Notice: No execution mode specified. Defaulting to safe --dry-run.\n"
            "Use --execute-upload to perform actual upload, or --prepare-manifest to view manifest.\n"
        )
        args.dry_run = True

    try:
        if args.prepare_manifest or args.save_prepared_manifest:
            hashes = verify_local_artifacts()
            manifest_data = prepare_manifest_data(
                repo_id=args.repo_id,
                revision=None,  # Leave unconfigured until actual upload commit exists
                computed_hashes=hashes,
            )
            manifest_json = json.dumps(manifest_data, indent=2)

            if args.prepare_manifest:
                print("=== Prepared Manifest ===")
                print(manifest_json)

            if args.save_prepared_manifest:
                out_path = Path(args.save_prepared_manifest).resolve()
                protected_manifest = (Path.cwd() / "configs" / "artifact_manifest.json").resolve()
                if out_path == protected_manifest:
                    raise ValueError(
                        "Refusing to overwrite authoritative 'configs/artifact_manifest.json'. "
                        "The manifest must not be overwritten automatically."
                    )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(manifest_json + "\n")
                print(f"[OK] Prepared manifest written to '{args.save_prepared_manifest}'.")

            if not args.execute_upload and not args.dry_run:
                return

        if args.dry_run:
            upload_artifacts(repo_id=args.repo_id, dry_run=True)
            return

        if args.execute_upload:
            result = upload_artifacts(
                repo_id=args.repo_id,
                commit_message=args.commit_message,
                dry_run=False,
            )
            print("\nUpload successfully completed!")
            print(f"Commit SHA: {result['commit_sha']}")

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        cleaned = sanitize_error_message(str(e))
        print(f"Error: {cleaned}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        cleaned = sanitize_error_message(str(e))
        print(f"Unexpected error: {cleaned}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
