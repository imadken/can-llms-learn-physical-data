"""Elapsed accelerator-job accounting and a hard deadline for owned work only."""

import json
import signal
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from effect_pretraining.schedule import ResourceBudget

_ACTIVE_RESERVATION: Path | None = None


class BudgetExpired(RuntimeError):
    pass


def read_budget(path: Path) -> ResourceBudget:
    budget = (
        ResourceBudget.model_validate_json(path.read_text()) if path.exists() else ResourceBudget()
    )
    marker = path.parent / "active_accelerator_job.json"
    if marker.exists():
        active = json.loads(marker.read_text())
        budget.accelerator_seconds_used += max(0, time.monotonic() - active["monotonic_started"])
    return budget


def begin_unavailable_wait(artifacts: Path) -> None:
    path = artifacts / "resource_budget.json"
    marker = artifacts / "active_accelerator_job.json"
    if marker.exists():
        raise RuntimeError("Another process holds the campaign reservation or needs recovery")
    budget = read_budget(path)
    if budget.unavailable_wait_started_utc is None:
        budget.unavailable_wait_started_utc = datetime.now(UTC)
        path.write_text(budget.model_dump_json(indent=2) + "\n")


def finish_unavailable_wait(artifacts: Path) -> None:
    """Called only for an explicit recovery after the machine prerequisite is addressed."""
    path = artifacts / "resource_budget.json"
    budget = read_budget(path)
    if budget.unavailable_wait_started_utc is not None:
        budget.unavailable_wait_seconds += (
            datetime.now(UTC) - budget.unavailable_wait_started_utc
        ).total_seconds()
        budget.unavailable_wait_started_utc = None
        path.write_text(budget.model_dump_json(indent=2) + "\n")


@contextmanager
def accelerator_job(artifacts: Path, name: str):
    global _ACTIVE_RESERVATION
    if _ACTIVE_RESERVATION is not None:
        if artifacts.resolve() != _ACTIVE_RESERVATION:
            raise RuntimeError("A reservation cannot silently account to another campaign")
        phase_start = time.monotonic()
        phase_status = "complete"
        try:
            yield read_budget(artifacts / "resource_budget.json")
        except BaseException:
            phase_status = "interrupted_or_fault"
            raise
        finally:
            with (artifacts / "accelerator_phases.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "name": name,
                            "elapsed_phase_seconds": time.monotonic() - phase_start,
                            "status": phase_status,
                            "included_in_parent_reservation": True,
                        }
                    )
                    + "\n"
                )
        return
    path = artifacts / "resource_budget.json"
    marker = artifacts / "active_accelerator_job.json"
    if marker.exists():
        raise RuntimeError("Another process holds the campaign reservation or needs recovery")
    budget = read_budget(path)
    if budget.unavailable_wait_started_utc is not None:
        raise BudgetExpired("Machine unavailability is pending explicit prerequisite recovery")
    started = datetime.now(UTC)
    if budget.profiling_started_utc is None:
        budget.profiling_started_utc = started
    remaining = min(
        budget.accelerator_limit_seconds - budget.accelerator_seconds_used,
        budget.execution_limit_seconds
        - (started - budget.profiling_started_utc).total_seconds()
        + budget.unavailable_wait_seconds,
        (budget.cutoff_utc - started).total_seconds(),
    )
    if remaining <= 180:
        raise BudgetExpired("Checkpoint/report reserve has been reached")
    path.write_text(budget.model_dump_json(indent=2) + "\n")
    clock = time.monotonic()
    marker.write_text(
        json.dumps({"name": name, "started_utc": started.isoformat(), "monotonic_started": clock})
        + "\n"
    )
    _ACTIVE_RESERVATION = artifacts.resolve()
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
        _ACTIVE_RESERVATION = None
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
