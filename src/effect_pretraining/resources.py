"""Elapsed accelerator-job accounting and a hard deadline for owned work only."""

import json
import signal
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from effect_pretraining.schedule import ResourceBudget


class BudgetExpired(RuntimeError):
    pass


def read_budget(path: Path) -> ResourceBudget:
    return (
        ResourceBudget.model_validate_json(path.read_text()) if path.exists() else ResourceBudget()
    )


@contextmanager
def accelerator_job(artifacts: Path, name: str):
    path = artifacts / "resource_budget.json"
    budget = read_budget(path)
    started = datetime.now(UTC)
    if budget.profiling_started_utc is None:
        budget.profiling_started_utc = started
    remaining = min(
        budget.accelerator_limit_seconds - budget.accelerator_seconds_used,
        budget.execution_limit_seconds - (started - budget.profiling_started_utc).total_seconds(),
        (budget.cutoff_utc - started).total_seconds(),
    )
    if remaining <= 180:
        raise BudgetExpired("Checkpoint/report reserve has been reached")
    path.write_text(budget.model_dump_json(indent=2) + "\n")
    marker = artifacts / "active_accelerator_job.json"
    if marker.exists():
        raise RuntimeError("An earlier accelerator job is active or needs accounting recovery")
    marker.write_text(json.dumps({"name": name, "started_utc": started.isoformat()}) + "\n")
    clock = time.monotonic()
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expire(signum, frame):
        raise BudgetExpired("Hard resource deadline reached; remaining time reserved for evidence")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, remaining - 180)
    status = "complete"
    try:
        yield budget
    except BaseException:
        status = "interrupted_or_fault"
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        elapsed = time.monotonic() - clock
        budget.accelerator_seconds_used += elapsed
        path.write_text(budget.model_dump_json(indent=2) + "\n")
        with (artifacts / "accelerator_jobs.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "name": name,
                        "started_utc": started.isoformat(),
                        "finished_utc": datetime.now(UTC).isoformat(),
                        "elapsed_job_seconds": elapsed,
                        "status": status,
                    }
                )
                + "\n"
            )
        marker.unlink(missing_ok=True)
