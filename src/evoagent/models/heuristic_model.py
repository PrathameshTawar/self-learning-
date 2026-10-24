"""
Deterministic heuristic and memory-guided coding model for testing and baselines.
"""

from evoagent.models.base import IModelBackend, ModelPrompt, ModelResponse


class HeuristicCodingModel(IModelBackend):
    """
    Heuristic rule-based model that can solve common boundary bugs and
    apply retrieved memory patterns deterministically.
    Useful for offline testing, CI/CD, and reproducible benchmarking.
    """

    def __init__(self, model_name: str = "heuristic-v1") -> None:
        self.model_name = model_name

    def generate_repair_proposal(self, prompt: ModelPrompt) -> ModelResponse:
        """Analyze failure trace and file content to propose a targeted patch."""
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

        # Default fallback
        return ModelResponse(
            hypothesis="Unable to determine precise failure cause with heuristics.",
            proposed_diff="",
            explanation="No matching heuristic rule found for failure trace.",
            model_name=self.model_name,
        )

    def _fix_index_error_empty_input(self, target_file: str, content: str) -> str:
        """Generate unified diff for empty string / list index access."""
        # Look for return parts[0].strip() or return parts[0]
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
