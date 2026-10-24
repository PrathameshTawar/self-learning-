"""
Base interface and dataclasses for pluggable coding model backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ModelPrompt(BaseModel):
    """Structured context passed to a repair model."""

    task_id: str
    issue_description: str
    failure_trace: str
    target_file: str
    file_content: str
    retrieved_lessons: List[str] = Field(default_factory=list)
    attempt_number: int = 1
    system_instruction: Optional[str] = None


class ModelResponse(BaseModel):
    """Structured response from a repair model."""

    hypothesis: str
    proposed_diff: str
    explanation: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_name: str = "unknown"


class IModelBackend(ABC):
    """Protocol for model inference backends."""

    @abstractmethod
    def generate_repair_proposal(self, prompt: ModelPrompt) -> ModelResponse:
        """Generate a repair plan and unified diff patch for a given bug."""
        pass
