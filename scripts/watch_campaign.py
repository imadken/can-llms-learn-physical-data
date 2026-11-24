"""One-shot hard limit guard for the currently owned campaign process."""

import argparse
import json
import os
import signal
import time
from datetime import UTC, datetime
from pathlib import Path


def owned_campaign(root: Path) -> int:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            args = (entry / "cmdline").read_bytes().split(b"\0")
            if args[1:3] != [b"-m", b"effect_pretraining.campaign"]:
                continue
            if not Path(os.readlink(entry / "exe")).name.startswith("python"):
                continue
            if Path(os.readlink(entry / "cwd")).resolve() == root:
                matches.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if len(matches) != 1:
        raise RuntimeError(f"Expected one owned campaign process, found {matches}")
    return matches[0]


def watch(root: Path, artifacts: Path) -> None:
    pid = owned_campaign(root)
    marker_path = artifacts / "active_accelerator_job.json"
    marker = json.loads(marker_path.read_text())
    report = {
        "status": "watching",
        "owned_pid": pid,
        "reservation_monotonic_started": marker["monotonic_started"],
        "scope": "current owned campaign only; exits when it ends; no scheduled continuation",
    }
    output = root / "audit/resource_watchdog.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    while Path(f"/proc/{pid}").exists() and marker_path.exists():
        current = json.loads(marker_path.read_text())
        if current["monotonic_started"] != marker["monotonic_started"]:
            raise RuntimeError("Reservation identity changed")
        budget = json.loads((artifacts / "resource_budget.json").read_text())
        now = datetime.now(UTC)
        used = budget["accelerator_seconds_used"] + time.monotonic() - marker["monotonic_started"]
        wall = (
            now - datetime.fromisoformat(budget["profiling_started_utc"].replace("Z", "+00:00"))
        ).total_seconds()
        wall -= budget.get("unavailable_wait_seconds", 0)
        remaining = min(
            budget["accelerator_limit_seconds"] - used,
            budget["execution_limit_seconds"] - wall,
            (
                datetime.fromisoformat(budget["cutoff_utc"].replace("Z", "+00:00")) - now
            ).total_seconds(),
        )
        if remaining <= 150:
            # Back up the in-process deadline, which reserves 180 seconds for checkpointing.
            os.kill(pid, signal.SIGINT)
            report.update(
                status="limit_interrupt_sent",
                remaining_seconds=remaining,
                interrupted_utc=now.isoformat(),
            )
            output.write_text(json.dumps(report, indent=2) + "\n")
            for _ in range(10):
                if not Path(f"/proc/{pid}").exists():
                    break
                time.sleep(1)
            if Path(f"/proc/{pid}").exists():
                os.kill(pid, signal.SIGTERM)
                report["termination_required"] = True
            break
        time.sleep(5)
    if report["status"] == "watching":
        report["status"] = "campaign_ended_without_limit_intervention"
    report["finished_utc"] = datetime.now(UTC).isoformat()
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    watch(Path.cwd().resolve(), args.artifacts.resolve())
