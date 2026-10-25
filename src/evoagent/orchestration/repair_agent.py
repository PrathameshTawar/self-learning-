"""
Repair agent coordinating model proposal generation and patch formatting.
"""

from evoagent.domain.models import PatchProposal
from evoagent.models.base import IModelBackend, ModelPrompt, ModelResponse
from evoagent.tools.patch_tools import PatchTool


class RepairAgent:
    """
    Invokes the model backend with the planned prompt, formats unified diffs,
    and performs pre-flight validation before sandbox application.
    """

    def __init__(self, model_backend: IModelBackend, patch_tool: PatchTool) -> None:
        self.model_backend = model_backend
        self.patch_tool = patch_tool

    def generate_candidate_patch(self, prompt: ModelPrompt) -> PatchProposal:
        """Query model backend and construct validated PatchProposal."""
        response: ModelResponse = self.model_backend.generate_repair_proposal(prompt)

        # Count added and removed lines
        diff_lines = response.proposed_diff.splitlines()
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))

        proposal = PatchProposal(
            target_file=prompt.target_file,
            diff_unified=response.proposed_diff,
            rationale=response.hypothesis or response.explanation,
            lines_added=added,
            lines_removed=removed,
            model_output=response.explanation,
        )

        # Pre-flight format validation
        is_valid, error_msg = self.patch_tool.validate_patch(proposal)
        proposal.is_valid_format = is_valid
        proposal.validation_error = error_msg

        return proposal
