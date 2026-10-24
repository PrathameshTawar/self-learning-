"""
Test mock heuristic coding model for unit testing and offline pipeline validation.
IMPORTANT: This class is solely an offline test mock / fixture. It is NOT intended
for production evaluation or real-world bug benchmark scoring. For real repair evaluation,
use OpenAICompatibleModel paired with an LLM (e.g., Qwen2.5-Coder via Ollama/vLLM).
"""

from evoagent.models.base import IModelBackend, ModelPrompt, ModelResponse


class HeuristicCodingModel(IModelBackend):
    """
    Mock rule-based model designed specifically for offline integration tests and synthetic CI fixtures.
    Simulates deterministic repair proposals for test fixtures without requiring an active LLM endpoint.
    """

    def __init__(self, model_name: str = "mock-heuristic-v1") -> None:
        self.model_name = model_name

    def generate_repair_proposal(self, prompt: ModelPrompt) -> ModelResponse:
        """Analyze failure trace and file content to propose a targeted mock patch."""
        # Check if retrieved memories contain a verified patch pattern
        for lesson in prompt.retrieved_lessons:
            if "empty" in lesson.lower() and "IndexError" in prompt.failure_trace:
                diff = self._fix_index_error_empty_input(prompt.target_file, prompt.file_content)
                if diff:
                    return ModelResponse(
                        hypothesis="Memory-guided repair: empty sequence causes IndexError, guard access.",
                        proposed_diff=diff,
                        explanation=f"Reused pattern from past lesson: {lesson}",
                        model_name=self.model_name,
                    )

        # Handle IndexError from empty input / sequence slicing
        if "IndexError" in prompt.failure_trace or "empty input" in prompt.issue_description.lower():
            diff = self._fix_index_error_empty_input(prompt.target_file, prompt.file_content)
            if diff:
                return ModelResponse(
                    hypothesis="Function attempts indexing without checking if the collection is non-empty.",
                    proposed_diff=diff,
                    explanation="Guard against empty collection before accessing index 0.",
                    model_name=self.model_name,
                )

        # Handle ZeroDivisionError
        if "ZeroDivisionError" in prompt.failure_trace:
            diff = self._fix_zero_division(prompt.target_file, prompt.file_content)
            if diff:
                return ModelResponse(
                    hypothesis="Denominator can be zero leading to ZeroDivisionError.",
                    proposed_diff=diff,
                    explanation="Check for zero before division.",
                    model_name=self.model_name,
                )

        # Handle AttributeError on NoneType input
        if "AttributeError" in prompt.failure_trace or "NoneType" in prompt.failure_trace:
            diff = self._fix_none_attribute_error(prompt.target_file, prompt.file_content)
            if diff:
                return ModelResponse(
                    hypothesis="Argument can be None, causing AttributeError on method call.",
                    proposed_diff=diff,
                    explanation="Guard against None input before invoking string methods.",
                    model_name=self.model_name,
                )

        # Default fallback
        return ModelResponse(
            hypothesis="Unable to determine precise failure cause with heuristics.",
            proposed_diff="",
            explanation="No matching heuristic rule found for failure trace.",
            model_name=self.model_name,
        )

    def _fix_index_error_empty_input(self, target_file: str, content: str) -> str:
        """Generate unified diff for empty string / list index access."""
        if "return parts[0].strip()" in content:
            return (
                f"--- a/{target_file}\n"
                f"+++ b/{target_file}\n"
                f"@@ -1,3 +1,3 @@\n"
                f"-    return parts[0].strip()\n"
                f"+    return parts[0].strip() if parts else \"\""
            )
        elif "return parts[0]" in content:
            return (
                f"--- a/{target_file}\n"
                f"+++ b/{target_file}\n"
                f"@@ -1,3 +1,3 @@\n"
                f"-    return parts[0]\n"
                f"+    return parts[0] if parts else \"\""
            )
        return ""

    def _fix_zero_division(self, target_file: str, content: str) -> str:
        """Generate unified diff for division by zero."""
        if "return a / b" in content:
            return (
                f"--- a/{target_file}\n"
                f"+++ b/{target_file}\n"
                f"@@ -1,3 +1,3 @@\n"
                f"-    return a / b\n"
                f"+    return a / b if b != 0 else 0"
            )
        return ""

    def _fix_none_attribute_error(self, target_file: str, content: str) -> str:
        """Generate unified diff for None input attribute access."""
        if "cleaned = raw_text.strip()" in content:
            return (
                f"--- a/{target_file}\n"
                f"+++ b/{target_file}\n"
                f"@@ -1,3 +1,4 @@\n"
                f"+    if raw_text is None:\n"
                f"+        return \"\"\n"
                f"     cleaned = raw_text.strip()"
            )
        return ""
