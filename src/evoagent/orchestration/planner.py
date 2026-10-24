"""
Task planner responsible for forming hypotheses from failure traces, memory lessons,
and strategy gene configurations.
"""

from pathlib import Path
import re
from typing import List, Optional
from evoagent.domain.models import Task, TestResult
from evoagent.memory.retrieval import ExperienceRetriever
from evoagent.models.base import ModelPrompt
from evoagent.tools.search_tools import CodeSearcher


class TaskPlanner:
    """
    Analyzes error stack traces, queries episodic memory for prior lessons,
    locates candidate failure files, and prepares model prompts according to strategy genes.
    """

    def __init__(self, retriever: ExperienceRetriever) -> None:
        self.retriever = retriever

    def extract_failure_signature(self, test_result: TestResult) -> str:
        """Extract primary exception type or message from test output."""
        output = f"{test_result.stdout}\n{test_result.stderr}"
        match = re.search(r"([A-Za-z]+Error|Exception): (.*)", output)
        if match:
            return f"{match.group(1)}: {match.group(2).strip()}"

        # Fallback to last error line
        for line in reversed(output.splitlines()):
            if "Error" in line or "FAILED" in line:
                return line.strip()

        return "Unspecified test failure"

    def locate_target_file(self, task: Task, test_result: TestResult, searcher: CodeSearcher) -> str:
        """Determine which source file contains the defect."""
        if "target_file" in task.metadata:
            return task.metadata["target_file"]

        # Parse from test stdout traceback
        output = f"{test_result.stdout}\n{test_result.stderr}"
        matches = re.findall(r'File "([^"]+\.py)", line \d+', output)
        for m in reversed(matches):
            p = Path(m)
            try:
                rel = p.relative_to(task.repository_path).as_posix()
                if not rel.startswith("tests/") and not rel.startswith("test_"):
                    return rel
            except Exception:
                if not m.startswith("test") and not "/test" in m:
                    return Path(m).as_posix()

        # Fallback: find source files matching test name
        files = searcher.list_files(".py")
        non_test_files = [f for f in files if "test" not in f]
        return non_test_files[0] if non_test_files else "src/main.py"

    def _get_prompt_style_instruction(self, style: str) -> str:
        """Return system instruction based on the prompt_style strategy gene."""
        style_clean = str(style).lower()
        if "chain_of_thought" in style_clean:
            return "Reasoning Mode: Chain-of-Thought. Step-by-step reasoning required: 1. Identify failure trigger. 2. Trace code flow. 3. Formulate fix. 4. Produce SEARCH/REPLACE block."
        elif "defensive" in style_clean:
            return "Reasoning Mode: Defensive Boundary. Focus strictly on defensive edge cases: check empty collections, None guards, zero denominators, and off-by-one indices."
        elif "hypothesis" in style_clean:
            return "Reasoning Mode: Hypothesis First. Explicitly state a falsifiable root-cause hypothesis before formulating code modifications."
        return "Reasoning Mode: Minimal Diff. Keep changes concise, minimal, and surgical to resolve the failure."

    def plan_repair(
        self,
        task: Task,
        pre_test: TestResult,
        searcher: CodeSearcher,
        attempt: int = 1,
        previous_attempt_patch: Optional[str] = None,
        previous_attempt_error: Optional[str] = None,
        prompt_style: str = "minimal_diff",
        retrieval_policy: str = "signature_match",
        retrieval_top_k: int = 3,
        enable_multi_file_context: bool = True,
    ) -> ModelPrompt:
        """Construct prompt incorporating strategy genes, context, retry feedback, and memory lessons."""
        signature = self.extract_failure_signature(pre_test)
        target_file = self.locate_target_file(task, pre_test, searcher)

        # 1. Assemble context respecting enable_multi_file_context gene
        content = ""
        if enable_multi_file_context:
            try:
                from evoagent.intelligence.context_builder import MultiFileContextBuilder
                builder = MultiFileContextBuilder(task.repository_path)
                ctx = builder.build_context(
                    primary_file=target_file,
                    failure_trace=f"{pre_test.stdout}\n{pre_test.stderr}",
                )
                content = ctx["formatted_context"]
            except Exception:
                try:
                    content = searcher.read_file(target_file)
                except Exception:
                    content = ""
        else:
            try:
                content = searcher.read_file(target_file)
            except Exception:
                content = ""

        # 2. Retrieve verified lessons respecting retrieval_policy & retrieval_top_k genes
        lessons = self.retriever.retrieve_relevant_lessons(
            failure_signature=signature,
            issue_description=task.issue_description,
            failure_trace=f"{pre_test.stdout}\n{pre_test.stderr}",
            max_results=retrieval_top_k,
            policy=retrieval_policy,
        )

        style_instruction = self._get_prompt_style_instruction(prompt_style)

        return ModelPrompt(
            task_id=task.task_id,
            issue_description=task.issue_description,
            failure_trace=f"{pre_test.stdout}\n{pre_test.stderr}",
            target_file=target_file,
            file_content=content,
            retrieved_lessons=lessons,
            attempt_number=attempt,
            previous_attempt_patch=previous_attempt_patch,
            previous_attempt_error=previous_attempt_error,
            system_instruction=style_instruction,
        )
