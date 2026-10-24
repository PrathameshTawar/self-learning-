"""
Task planner responsible for forming hypotheses from failure traces and memory lessons.
"""

import re
from pathlib import Path
from typing import List, Optional
from evoagent.domain.models import Task, TestResult
from evoagent.memory.retrieval import ExperienceRetriever
from evoagent.models.base import ModelPrompt
from evoagent.tools.search_tools import CodeSearcher


class TaskPlanner:
    """
    Analyzes error stack traces, queries episodic memory for prior lessons,
    locates candidate failure files, and prepares model prompts.
    """

    def __init__(self, retriever: ExperienceRetriever) -> None:
        self.retriever = retriever

    def extract_failure_signature(self, test_result: TestResult) -> str:
        """Extract primary exception type or message from test output."""
        output = f"{test_result.stdout}\n{test_result.stderr}"
        match = re.search(r"([A-Za-z]+Error|Exception): (.*)", output)
        if match:
            return f"{match.group(1)}: {match.group(2).strip()}"

        # Fallback to last line
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
        # Look for File ".../something.py", line ...
        matches = re.findall(r'File "([^"]+\.py)", line \d+', output)
        for m in reversed(matches):
            p = Path(m)
            # Find relative path within repository
            try:
                rel = p.relative_to(task.repository_path).as_posix()
                if not rel.startswith("tests/") and not rel.startswith("test_"):
                    return rel
            except Exception:
                # If path was relative in output
                if not m.startswith("test") and not "/test" in m:
                    return Path(m).as_posix()

        # Fallback: find source files matching test name
        files = searcher.list_files(".py")
        non_test_files = [f for f in files if "test" not in f]
        return non_test_files[0] if non_test_files else "src/main.py"

    def plan_repair(
        self,
        task: Task,
        pre_test: TestResult,
        searcher: CodeSearcher,
        attempt: int = 1,
    ) -> ModelPrompt:
        """Construct prompt with multi-file AST context, failure trace, and memory lessons."""
        signature = self.extract_failure_signature(pre_test)
        target_file = self.locate_target_file(task, pre_test, searcher)

        # Assemble multi-file context via AST symbol navigation
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

        # Retrieve verified lessons from past experiences
        lessons = self.retriever.retrieve_relevant_lessons(
            failure_signature=signature,
            issue_description=task.issue_description,
            max_results=3,
        )

        return ModelPrompt(
            task_id=task.task_id,
            issue_description=task.issue_description,
            failure_trace=f"{pre_test.stdout}\n{pre_test.stderr}",
            target_file=target_file,
            file_content=content,
            retrieved_lessons=lessons,
            attempt_number=attempt,
        )
