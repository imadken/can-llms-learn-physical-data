"""Run with the verified existing environment and record complete commands."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = json.loads((root / "resolved_environment.json").read_text())["runtime_paths"]
    dependencies = root / ".runtime/dependencies"
    dependencies.mkdir(parents=True, exist_ok=True)
    for name, target in {
        "lerobot": Path(paths["lerobot_source"]),
        "libero": Path(paths["libero_source"]) / "libero",
    }.items():
        link = dependencies / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    config_dir = root / ".runtime/libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    libero = Path(paths["libero_source"]) / "libero/libero"
    (config_dir / "config.yaml").write_text(
        "\n".join(
            f"{key}: {value}"
            for key, value in {
                "benchmark_root": libero,
                "bddl_files": libero / "bddl_files",
                "init_states": libero / "init_files",
                "datasets": libero.parent / "datasets",
                "assets": libero / "assets",
            }.items()
        )
        + "\n"
    )
    environment = os.environ.copy()
    overrides = {
        "PYTHONPATH": f"{root / 'src'}:{dependencies}",
        "LIBERO_CONFIG_PATH": str(config_dir),
        "MUJOCO_GL": "osmesa",
        "LD_LIBRARY_PATH": paths["osmesa_library_dir"],
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
        "MPLBACKEND": "Agg",
    }
    environment.update(overrides)
    command = ["uv", "run", "--no-project", "--no-config", paths["python"], *sys.argv[1:]]
    audit = root / "audit"
    audit.mkdir(exist_ok=True)
    index = len(list(audit.glob("command_*.json")))
    record_path = audit / f"command_{index:03d}.json"
    record = {
        "started_utc": datetime.now(UTC).isoformat(),
        "cwd": str(root),
        "argv": command,
        "environment_overrides": overrides,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "log_path": str(audit / f"command_{index:03d}.log"),
    }
    stdin_source = sys.stdin.read() if sys.argv[1:] == ["-"] else None
    if stdin_source is not None:
        source_path = audit / f"command_{index:03d}.stdin.py"
        source_path.write_text(stdin_source)
        record["stdin_source_path"] = str(source_path)
        record["stdin_source_sha256"] = hashlib.sha256(stdin_source.encode()).hexdigest()
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    with Path(record["log_path"]).open("w") as logfile:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_source is not None else None,
            text=True,
        )
        if stdin_source is not None:
            process.stdin.write(stdin_source)
            process.stdin.close()
        for line in process.stdout:
            logfile.write(line)
            logfile.flush()
            print(line, end="", flush=True)
        returncode = process.wait()
    record.update(finished_utc=datetime.now(UTC).isoformat(), returncode=returncode)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
