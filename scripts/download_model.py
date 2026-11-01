import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
SHA256_HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def sanitize_error_message(message: str) -> str:
    """Masks potential authentication tokens and credentials from error strings."""
    sanitized = re.sub(r"hf_[a-zA-Z0-9]{8,}", "hf_***REDACTED***", message)
    sanitized = re.sub(r"(token=)[^\s&]+", r"\1***REDACTED***", sanitized)
    sanitized = re.sub(r"(Bearer\s+)[^\s]+", r"\1***REDACTED***", sanitized)
    return sanitized


def compute_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Computes the SHA-256 hex digest of a local file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def load_manifest(manifest_path: str) -> dict:
    """Loads and validates the artifact manifest JSON file."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Artifact manifest not found at: {manifest_path}")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse artifact manifest at {manifest_path}: {e}") from e

    if not isinstance(manifest, dict):
        raise ValueError(f"Artifact manifest must be a JSON object, got {type(manifest).__name__}")

    return manifest


def is_immutable_revision(revision: str | None) -> bool:
    """
    Returns True only if the revision is an explicit, immutable full 40-character
    hexadecimal Git commit SHA without leading or trailing whitespace.
    Rejects short SHAs, branches, tags, whitespace, and malformed strings.
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


def validate_remote_filename(filename: str | None) -> str:
    """
    Validates that a remote filename is a safe relative path without traversal.
    Rejects None, empty, absolute paths, drive letters, and parent directory traversal.
    """
    if not filename or not isinstance(filename, str) or not filename.strip():
        raise ValueError("Remote filename is unconfigured or empty.")

    clean_fn = filename.strip()
    if os.path.isabs(clean_fn) or clean_fn.startswith(("/", "\\")):
        raise ValueError(f"Unsafe remote filename '{filename}': absolute paths are not permitted.")

    if re.match(r"^[a-zA-Z]:", clean_fn):
        raise ValueError(f"Unsafe remote filename '{filename}': drive paths are not permitted.")

    parts = re.split(r"[/\\]+", clean_fn)
    if any(part == ".." for part in parts):
        raise ValueError(f"Unsafe remote filename '{filename}': parent traversal ('..') is not permitted.")

    return clean_fn


def validate_and_resolve_local_path(local_path: str | None, project_root: str) -> str:
    """
    Validates that a local destination path is relative and strictly confined within project_root.
    Rejects None, empty, absolute paths, drive letters, and parent directory traversal.
    Resolves symlinks (via Path.resolve(strict=False)) to ensure the real target stays within the resolved project root.
    Returns the resolved absolute path safely within project_root.
    """
    if not local_path or not isinstance(local_path, str) or not local_path.strip():
        raise ValueError("Local destination path is unconfigured or empty.")

    clean_path = local_path.strip()

    if os.path.isabs(clean_path) or clean_path.startswith(("/", "\\")):
        raise ValueError(f"Unsafe local destination '{local_path}': absolute paths are not permitted.")

    if re.match(r"^[a-zA-Z]:", clean_path):
        raise ValueError(f"Unsafe local destination '{local_path}': drive paths are not permitted.")

    parts = re.split(r"[/\\]+", clean_path)
    if any(part == ".." for part in parts):
        raise ValueError(f"Unsafe local destination '{local_path}': parent traversal ('..') is not permitted.")

    root_path = Path(project_root).resolve()
    target_path = (root_path / clean_path).resolve(strict=False)

    try:
        target_path.relative_to(root_path)
    except ValueError:
        raise ValueError(
            f"Unsafe local destination '{local_path}': resolves outside project root ({root_path})."
        )

    if target_path == root_path:
        raise ValueError(f"Unsafe local destination '{local_path}': cannot point to project root itself.")

    return str(target_path)


def download_and_verify_artifact(
    repo_id: str,
    revision: str,
    remote_filename: str,
    local_path: str,
    expected_hash: str,
    project_root: str | None = None,
) -> bool:
    """
    Downloads a single artifact to a temporary directory, validates its SHA-256
    against the expected hash, and safely places it into the target destination.

    If the local file already exists and matches expected_hash, it is reused without downloading.
    """
    from huggingface_hub import hf_hub_download

    effective_root = os.path.abspath(project_root) if project_root else os.path.abspath(os.getcwd())
    validate_remote_filename(remote_filename)
    target_file = validate_and_resolve_local_path(local_path, effective_root)

    if not is_valid_sha256(expected_hash):
        raise ValueError(
            f"Expected checksum {expected_hash!r} is invalid. "
            "Checksum must be exactly 64 hexadecimal characters."
        )
    expected_hash_clean = expected_hash.lower()

    # Step 1: Check if an existing local file is already valid
    if os.path.isfile(target_file):
        try:
            existing_hash = compute_sha256(target_file)
            if existing_hash == expected_hash_clean:
                print(f"[OK] Reusing existing verified artifact at '{local_path}' (SHA-256 match).")
                return True
            else:
                print(
                    f"Notice: Existing file at '{local_path}' does not match expected SHA-256 "
                    f"(found: {existing_hash[:8]}..., expected: {expected_hash_clean[:8]}...). "
                    "Downloading verified artifact from Hub."
                )
        except OSError as e:
            print(f"Warning: Could not read existing file '{local_path}': {e}. Downloading new copy.")

    # Step 2: Download into an isolated temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            temp_download_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_filename,
                revision=revision,
                local_dir=temp_dir,
            )
        except Exception as e:
            cleaned_error = sanitize_error_message(str(e))
            raise RuntimeError(
                f"Failed to download '{remote_filename}' from repository '{repo_id}' "
                f"(revision: '{revision}'): {cleaned_error}"
            ) from None

        if not os.path.isfile(temp_download_path):
            raise FileNotFoundError(
                f"Downloaded file '{remote_filename}' was not found at expected temporary path."
            )

        # Step 3: Verify SHA-256 before touching target location
        downloaded_hash = compute_sha256(temp_download_path)
        if downloaded_hash != expected_hash_clean:
            raise ValueError(
                f"SHA-256 verification failed for '{remote_filename}'. "
                f"Expected {expected_hash_clean}, got {downloaded_hash}. "
                "Download rejected and temporary file discarded; target file was NOT overwritten."
            )

        # Step 4: Safely place verified file into target destination
        target_dir = os.path.dirname(target_file)
        os.makedirs(target_dir, exist_ok=True)

        temp_target = os.path.join(target_dir, f".{os.path.basename(target_file)}.tmp")
        try:
            shutil.copy2(temp_download_path, temp_target)
            os.replace(temp_target, target_file)
        finally:
            if os.path.exists(temp_target):
                try:
                    os.remove(temp_target)
                except OSError:
                    pass

    print(f"[OK] Verified and placed artifact: '{local_path}' (SHA-256: {expected_hash_clean[:8]}...)")
    return True


def download_model(
    model_name: str,
    manifest_path: str = "configs/artifact_manifest.json",
    project_root: str | None = None,
) -> bool:
    """
    Downloads and verifies all artifacts for a given model as defined in the manifest.
    Refuses execution if repository, revision, or SHA-256 hashes are unconfigured,
    or if paths/filenames violate safety and containment boundaries.
    """
    print(f"=== Inspecting Artifact Manifest for Model: {model_name} ===")

    effective_root = os.path.abspath(project_root) if project_root else os.path.abspath(os.getcwd())
    manifest = load_manifest(manifest_path)
    models = manifest.get("models", manifest)

    if model_name not in models:
        raise ValueError(
            f"Model '{model_name}' not defined in manifest. Available models: {list(models.keys())}"
        )

    model_entry = models[model_name]
    repo_id = model_entry.get("repo_id")
    revision = model_entry.get("revision")
    artifacts = model_entry.get("artifacts")

    # Guard 1: Repository must be configured
    if not repo_id or not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError(
            f"Model '{model_name}' has unconfigured repository ID ('repo_id': {repo_id!r}) in manifest. "
            "Downloader refuses to run until Hugging Face repository is configured."
        )

    # Guard 2: Revision must be configured and an immutable full 40-character commit SHA
    if not revision or not isinstance(revision, str) or not revision.strip():
        raise ValueError(
            f"Model '{model_name}' has unconfigured Hub revision ('revision': {revision!r}) in manifest. "
            "Downloader refuses to run until an immutable 40-character commit SHA is configured."
        )

    if not is_immutable_revision(revision):
        raise ValueError(
            f"Model '{model_name}' specifies invalid or mutable revision {revision!r}. "
            "Downloader requires an immutable full 40-character hexadecimal Git commit SHA without whitespace."
        )

    # Guard 3: Artifacts structure, filenames, checksums, and path confinement
    if not artifacts or not isinstance(artifacts, dict):
        raise ValueError(f"Model '{model_name}' has no artifacts defined in manifest.")

    for artifact_name, artifact_info in artifacts.items():
        if not isinstance(artifact_info, dict):
            raise ValueError(
                f"Artifact entry '{artifact_name}' for model '{model_name}' must be an object."
            )

        remote_fn = artifact_info.get("remote_filename")
        if not remote_fn or not isinstance(remote_fn, str) or not remote_fn.strip():
            raise ValueError(
                f"Artifact '{artifact_name}' for model '{model_name}' has unconfigured 'remote_filename' "
                "in manifest. Downloader refuses to run until remote filenames are specified."
            )
        validate_remote_filename(remote_fn)

        expected_hash = artifact_info.get("sha256")
        if not expected_hash or not isinstance(expected_hash, str) or not expected_hash.strip():
            raise ValueError(
                f"Artifact '{artifact_name}' for model '{model_name}' has unconfigured 'sha256' "
                "in manifest. Downloader refuses to run without authoritative checksum verification."
            )

        if not is_valid_sha256(expected_hash):
            raise ValueError(
                f"Artifact '{artifact_name}' for model '{model_name}' has malformed 'sha256' {expected_hash!r}. "
                "Checksum must be exactly 64 hexadecimal characters without whitespace."
            )

        local_path = artifact_info.get("local_path")
        if not local_path or not isinstance(local_path, str) or not local_path.strip():
            raise ValueError(
                f"Artifact '{artifact_name}' for model '{model_name}' has missing 'local_path' in manifest."
            )
        validate_and_resolve_local_path(local_path, effective_root)

    print(f"Repository: {repo_id}")
    print(f"Revision:   {revision}")
    print(f"Artifacts to verify/download: {len(artifacts)}")

    # All preflights and security boundaries passed; proceed to download and verify
    for artifact_name, artifact_info in artifacts.items():
        print(f"\nProcessing artifact: {artifact_name} ({artifact_info['local_path']})")
        download_and_verify_artifact(
            repo_id=repo_id,
            revision=revision,
            remote_filename=artifact_info["remote_filename"],
            local_path=artifact_info["local_path"],
            expected_hash=artifact_info["sha256"],
            project_root=effective_root,
        )

    print(f"\n[SUCCESS] All artifacts for {model_name} successfully verified and ready for use.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify canonical Tiny GPT models from Hugging Face Hub."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gpt1", "gpt2", "gpt3"],
        help="Canonical model to download (gpt1, gpt2, or gpt3).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="configs/artifact_manifest.json",
        help="Path to artifact manifest JSON file (default: configs/artifact_manifest.json).",
    )
    args = parser.parse_args()

    try:
        download_model(model_name=args.model, manifest_path=args.manifest)
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as e:
        cleaned = sanitize_error_message(str(e))
        print(f"Error: {cleaned}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        cleaned = sanitize_error_message(str(e))
        print(f"Unexpected error: {cleaned}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
