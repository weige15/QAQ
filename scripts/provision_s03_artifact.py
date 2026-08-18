#!/usr/bin/env python3
"""Provision the ignored S03-B artifact into this worktree.

The source is supplied explicitly because the multi-gigabyte artifact is not
tracked in Git. The source checkpoint is hashed before linking and the
worktree link is hashed again after provisioning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_s10h as runner


DESTINATION = ROOT / runner.PACKED_ARTIFACT
SOURCE_ENV = "QAQ_S03_ARTIFACT_SOURCE"
ANY_PRECISION_DESTINATION = ROOT / "third_party/any-precision-llm"
ANY_PRECISION_SOURCE_ENV = "QAQ_ANY_PRECISION_SOURCE"


class ArtifactProvisionError(RuntimeError):
    """The source or destination cannot satisfy the frozen artifact identity."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(path: Path) -> Path:
    if not path.is_dir():
        raise ArtifactProvisionError(f"artifact source directory is unavailable: {path}")
    checkpoint = path / "pytorch_model.bin"
    if not checkpoint.is_file():
        raise ArtifactProvisionError(f"artifact source checkpoint is unavailable: {checkpoint}")
    return checkpoint


def provision_artifact(source: Path, destination: Path = DESTINATION) -> dict[str, str]:
    """Link a verified artifact into ``destination`` without overwriting data."""

    source = source.expanduser().resolve()
    source_checkpoint = _checkpoint(source)
    source_digest = _sha256_file(source_checkpoint)
    if source_digest != runner.ARTIFACT_SHA256:
        raise ArtifactProvisionError(
            "artifact source SHA-256 does not match the frozen identity: "
            f"{source_digest} != {runner.ARTIFACT_SHA256}"
        )

    destination = destination.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination_checkpoint = _checkpoint(destination)
        destination_digest = _sha256_file(destination_checkpoint)
        if destination_digest != runner.ARTIFACT_SHA256:
            raise ArtifactProvisionError(
                "existing worktree artifact SHA-256 does not match the frozen identity: "
                f"{destination_digest} != {runner.ARTIFACT_SHA256}"
            )
    else:
        destination.symlink_to(source, target_is_directory=True)

    destination_digest = _sha256_file(_checkpoint(destination))
    if destination_digest != runner.ARTIFACT_SHA256:
        raise ArtifactProvisionError(
            "provisioned artifact SHA-256 does not match the frozen identity: "
            f"{destination_digest} != {runner.ARTIFACT_SHA256}"
        )
    return {
        "source": str(source),
        "destination": str(destination),
        "pytorch_model_sha256": destination_digest,
    }


def provision_any_precision(source: Path, destination: Path = ANY_PRECISION_DESTINATION) -> dict[str, str]:
    """Link a clean checkout of the pinned backend into the worktree."""

    source = source.expanduser().resolve()
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode != 0 or revision.stdout.strip() != runner.ANY_PRECISION_REVISION:
        raise ArtifactProvisionError("Any-Precision source is not at the frozen revision")
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain=v1"],
        capture_output=True,
        text=True,
        check=False,
    )
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise ArtifactProvisionError("Any-Precision source is dirty")

    destination = destination.expanduser()
    if destination.exists() and not destination.is_symlink():
        if any(destination.iterdir()):
            raise ArtifactProvisionError(f"non-empty Any-Precision destination will not be overwritten: {destination}")
        destination.rmdir()
    if not destination.exists() and not destination.is_symlink():
        destination.symlink_to(source, target_is_directory=True)

    actual = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if actual.returncode != 0 or actual.stdout.strip() != runner.ANY_PRECISION_REVISION:
        raise ArtifactProvisionError("provisioned Any-Precision checkout is not at the frozen revision")
    return {"source": str(source), "destination": str(destination), "revision": actual.stdout.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.environ[SOURCE_ENV]) if os.environ.get(SOURCE_ENV) else None,
        help=f"existing artifact directory (defaults to ${SOURCE_ENV})",
    )
    parser.add_argument(
        "--any-precision-source",
        type=Path,
        default=Path(os.environ[ANY_PRECISION_SOURCE_ENV]) if os.environ.get(ANY_PRECISION_SOURCE_ENV) else None,
        help=f"clean pinned Any-Precision checkout (defaults to ${ANY_PRECISION_SOURCE_ENV})",
    )
    args = parser.parse_args(argv)
    if args.source is None:
        parser.error(f"--source or ${SOURCE_ENV} is required")
    try:
        result = {"artifact": provision_artifact(args.source)}
        if args.any_precision_source is not None:
            result["any_precision"] = provision_any_precision(args.any_precision_source)
        print(json.dumps(result, sort_keys=True))
    except ArtifactProvisionError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
