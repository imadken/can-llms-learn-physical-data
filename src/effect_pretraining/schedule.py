"""Locked optimizer schedule and conservative resource admission."""

import math
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


def learning_rate(update: int, *, baseline: bool = False) -> float:
    total, warmup = (5000, 500) if baseline else (1000, 100)
    if not 1 <= update <= total:
        raise ValueError("Update outside the fixed pilot schedule")
    if update <= warmup:
        return 1e-4 * update / warmup
    phase = (update - warmup) / (total - warmup)
    return 2.5e-6 + (1e-4 - 2.5e-6) * (1 + math.cos(math.pi * phase)) / 2


class ResourceBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profiling_started_utc: datetime | None = None
    accelerator_seconds_used: float = Field(default=0, ge=0)
    unavailable_wait_seconds: float = Field(default=0, ge=0)
    unavailable_wait_started_utc: datetime | None = None
    accelerator_limit_seconds: int = 8 * 3600
    execution_limit_seconds: int = 12 * 3600
    cutoff_utc: datetime = datetime(2026, 11, 28, 17, tzinfo=UTC)

    def permits(self, now: datetime, projected_job_seconds: float, reserve_seconds: float) -> bool:
        if min(projected_job_seconds, reserve_seconds) < 0:
            raise ValueError("Projection and checkpoint/report reserve cannot be negative")
        if self.unavailable_wait_started_utc is not None:
            return False
        required = projected_job_seconds * 1.25 + reserve_seconds
        wall_remaining = (self.cutoff_utc - now).total_seconds()
        if self.profiling_started_utc is not None:
            wall_remaining = min(
                wall_remaining,
                self.execution_limit_seconds
                - (now - self.profiling_started_utc).total_seconds()
                + self.unavailable_wait_seconds,
            )
        return required <= min(
            self.accelerator_limit_seconds - self.accelerator_seconds_used, wall_remaining
        )


def common_acquisition_round(completed: dict[tuple[str, int], int]) -> int:
    expected = {(arm, seed) for arm in ("P", "K", "D") for seed in (101, 202)}
    if set(completed) != expected or any(x not in range(0, 1001, 250) for x in completed.values()):
        raise ValueError("Require all six paired fits at common checkpoint boundaries")
    return min(completed.values())
