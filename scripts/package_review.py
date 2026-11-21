"""Package the committed release plus five existing replay examples and external hashes."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from effect_pretraining.integrity import sha256_file, verify_release


def package(root: Path, artifacts: Path, implementation_commit: str) -> dict:
    verified = verify_release(root)
    results_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    owned_paths = [
        "src",
        "tests",
        "scripts",
        "docs",
        "audit",
        "research",
        "configs",
        "README.md",
        "RESEARCH_LOCK.json",
        "PACKAGE_MANIFEST.json",
        "resolved_environment.json",
        "pyproject.toml",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *owned_paths], cwd=root, text=True
    ).strip():
        raise RuntimeError("Commit auditable code/results before building the final bundle")
    destination = root / "bundles"
    destination.mkdir(exist_ok=True)
    resolved = json.loads((root / "resolved_environment.json").read_text())
    artifact_rows = []
    for path in sorted(artifacts.rglob("*")):
        if path.is_file():
            artifact_rows.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    paths = resolved["runtime_paths"]
    for directory in (Path(paths["checkpoint"]), Path(paths["vlm_assets"])):
        for path in sorted(directory.iterdir()):
            if path.is_file():
                artifact_rows.append(
                    {
                        "path": str(path.resolve()),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    runtime_inputs = [Path(item["path"]) for item in resolved["source_hashes"]]
    bank_path = root / "audit/downstream_evaluation_bank.json"
    if bank_path.exists():
        bank = json.loads(bank_path.read_text())
        for task in bank["tasks"]:
            for prefix in ("initial_states", "bddl"):
                path = Path(task[f"{prefix}_path"])
                if sha256_file(path) != task[f"{prefix}_sha256"]:
                    raise ValueError(f"Pinned downstream input changed: {path}")
                runtime_inputs.append(path)
    recorded_paths = {item["path"] for item in artifact_rows}
    for path in runtime_inputs:
        if str(path.resolve()) not in recorded_paths:
            artifact_rows.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
            recorded_paths.add(str(path.resolve()))
    original = root / "RSS_2027_Codex_Supervision_Package.zip"
    artifact_rows.append(
        {
            "path": str(original),
            "size_bytes": original.stat().st_size,
            "sha256": sha256_file(original),
        }
    )
    retained = destination / "retained_artifacts.json"
    retained.write_text(json.dumps(artifact_rows, indent=2) + "\n")
    examples = []
    for episode in range(90, 95):
        reference = f"ep{episode:03d}_t000_chunk0"
        files = [
            artifacts / "snapshots" / f"ep{episode:03d}_t000.npz",
            artifacts / "models" / f"episode_{episode:03d}.xml",
            artifacts / "replays" / f"{reference}_actions.npy",
            artifacts / "replays" / f"{reference}_repeat0.npz",
            artifacts / "replays" / f"{reference}_repeat1.npz",
        ]
        if not all(path.is_file() for path in files):
            raise FileNotFoundError("The five fixed first-episode replay examples must exist")
        examples.append(
            {
                "episode_id": episode,
                "offset": 0,
                "reference_chunk": 0,
                "files": [{"path": str(path), "sha256": sha256_file(path)} for path in files],
            }
        )
    delivery = {
        "created_utc": datetime.now(UTC).isoformat(),
        "repository_url": resolved["repository_url"],
        "branch": resolved["branch"],
        "implementation_commit": implementation_commit,
        "results_commit": results_commit,
        "unrelated_worktree_state_preserved": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).splitlines(),
        "remote_push_performed": False,
        "protected_hashes": verified,
        "retained_artifacts_manifest_sha256": sha256_file(retained),
        "examples": examples,
        "example_selection": "first five evaluation episode IDs, offset 0, chunk 0",
        "artifact_root": str(artifacts),
        "status_path": "repository/audit/status.json",
    }
    # Preserve the exact historical execution source as well as final review code.
    revisions = {implementation_commit}
    for name in ("collection_run.json", "residual_run.json"):
        record_path = artifacts / name
        if record_path.exists():
            revisions.add(json.loads(record_path.read_text())["code_commit"])
    for path in (root / "audit").glob("command_*.json"):
        command = json.loads(path.read_text())
        if "effect_pretraining.campaign" in command["argv"]:
            revisions.add(command["code_commit"])
    baseline_audit = root / "audit/actual_baseline_invariants.json"
    if baseline_audit.exists():
        revision = json.loads(baseline_audit.read_text()).get("code_commit")
        if revision:
            revisions.add(revision)
    execution_archives = {}
    for revision in sorted(revisions):
        content = subprocess.check_output(
            [
                "git",
                "archive",
                "--format=tar.gz",
                revision,
                "--",
                "src",
                "scripts",
                "tests",
                "pyproject.toml",
                "resolved_environment.json",
            ],
            cwd=root,
        )
        execution_archives[f"execution_source/{revision}.tar.gz"] = content
    delivery["execution_source_archives"] = {
        name: {"size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        for name, content in execution_archives.items()
    }
    delivery_path = destination / "delivery.json"
    delivery_path.write_text(json.dumps(delivery, indent=2) + "\n")
    archive = destination / "rss_stage1_review_20261121.tar.gz"
    if archive.exists():
        raise FileExistsError("Do not overwrite an existing review bundle silently")
    repository_tar = subprocess.check_output(
        ["git", "archive", "--format=tar", results_commit], cwd=root
    )
    with tarfile.open(archive, "w:gz") as output:
        with tarfile.open(fileobj=io.BytesIO(repository_tar), mode="r:") as source:
            for member in source:
                stream = source.extractfile(member) if member.isfile() else None
                member.name = "repository/" + member.name
                output.addfile(member, stream)
        output.add(delivery_path, arcname="delivery.json")
        output.add(retained, arcname="retained_artifacts.json")
        for name, content in execution_archives.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            output.addfile(member, io.BytesIO(content))
        for example in examples:
            for item in example["files"]:
                path = Path(item["path"])
                output.add(
                    path, arcname=f"examples/episode_{example['episode_id']:03d}/{path.name}"
                )
    checksum = sha256_file(archive)
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {archive.name}\n")
    # Read every archived regular member to ensure gzip/tar integrity before delivery.
    with tarfile.open(archive, "r:gz") as check:
        for member in check:
            if member.isfile():
                with check.extractfile(member) as stream:
                    while stream.read(1024 * 1024):
                        pass
    result = delivery | {
        "archive_path": str(archive),
        "archive_sha256": checksum,
        "external_checksum_path": str(checksum_path),
        "retained_artifact_count": len(artifact_rows),
        "size_bytes": archive.stat().st_size,
    }
    (destination / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"examples", "protected_hashes"}
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()
    package(Path.cwd(), args.artifacts.resolve(), args.implementation_commit)
