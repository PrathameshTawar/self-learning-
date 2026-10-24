"""
Task state definitions for EvoAgent finite state machine.
"""

from enum import Enum


class TaskState(str, Enum):
    """
    Explicit, auditable states for a coding task lifecycle:
    QUEUED -> ANALYZING -> PLANNED -> PATCHING -> EVALUATING -> SUCCEEDED / RETRYABLE_FAILURE / FAILED -> LEARNING_REVIEW -> COMPLETED
    """

    QUEUED = "QUEUED"
    ANALYZING = "ANALYZING"
    PLANNED = "PLANNED"
    PATCHING = "PATCHING"
    EVALUATING = "EVALUATING"
    SUCCEEDED = "SUCCEEDED"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    FAILED = "FAILED"
    INVALID_TASK = "INVALID_TASK"
    LEARNING_REVIEW = "LEARNING_REVIEW"
    COMPLETED = "COMPLETED"

    @property
    def is_terminal(self) -> bool:
        """Return True if the task has concluded its execution."""
        return self in (TaskState.COMPLETED, TaskState.FAILED, TaskState.INVALID_TASK)

    @property
    def is_successful(self) -> bool:
        """Return True if the state indicates a verified success."""
        return self in (TaskState.SUCCEEDED, TaskState.COMPLETED)
