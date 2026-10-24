"""
Data schemas for episodic memory and experience records.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class MemoryEpisodeRecord(BaseModel):
    """
    Episodic experience record storing task signatures, actions, outcomes, and provenance.
    Forms Layer A of the EvoAgent self-learning mechanism.
    """

    episode_id: str
    task_id: str
    issue_description: str
    failure_signature: str
    target_file: str
    root_cause_hypothesis: str
    applied_patch: str
    outcome: str = Field(..., description="'accepted' or 'rejected'")
    targeted_tests_passed: bool
    regression_tests_passed: bool
    lessons_learned: str
    model_version: str = "baseline-001"
    strategy_id: str = "default"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
