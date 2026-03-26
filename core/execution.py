"""Durable execution primitives: step persistence, retries, and resume.

This module provides the foundation for making agent runs resumable and
observable.  Phase 1 uses in-memory storage; swap ``RunStore`` for a
DB-backed implementation to enable cross-process resume and Temporal-style
durability later.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class StepRecord:
    step_index: int
    type: str  # "model_call" | "tool_execution"
    status: StepStatus = StepStatus.PENDING
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    input_snapshot: Optional[Dict[str, Any]] = None
    output_snapshot: Optional[Any] = None
    error: Optional[str] = None
    attempts: int = 0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    idempotency_key: Optional[str] = None


@dataclass
class RunRecord:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    agent_id: str = "default"
    profile: str = "default"
    status: StepStatus = StepStatus.PENDING
    steps: List[StepRecord] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def last_completed_step(self) -> int:
        completed = [s.step_index for s in self.steps if s.status == StepStatus.COMPLETED]
        return max(completed) if completed else -1


class RunStore:
    """In-memory run store.  Replace with DB-backed implementation for
    production durability."""

    def __init__(self) -> None:
        self._runs: Dict[str, RunRecord] = {}

    def save(self, run: RunRecord) -> None:
        run.updated_at = time.time()
        self._runs[run.run_id] = run

    def get(self, run_id: str) -> Optional[RunRecord]:
        return self._runs.get(run_id)

    def list_runs(self, agent_id: Optional[str] = None) -> List[RunRecord]:
        runs = list(self._runs.values())
        if agent_id:
            runs = [r for r in runs if r.agent_id == agent_id]
        return sorted(runs, key=lambda r: r.created_at, reverse=True)


@dataclass
class RetryPolicy:
    """Configurable retry policy for model and tool calls."""

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple = (TimeoutError, ConnectionError)

    def delay_for_attempt(self, attempt: int) -> float:
        delay = self.base_delay_seconds * (self.exponential_base**attempt)
        return min(delay, self.max_delay_seconds)
