"""
Task planner responsible for forming hypotheses from failure traces, memory lessons,
and strategy gene configurations.
"""

import ast
from pathlib import Path
import re
from typing import List, Optional
from evoagent.domain.models import Task, TestResult
from evoagent.memory.retrieval import ExperienceRetriever
from evoagent.models.base import ModelPrompt
from evoagent.tools.search_tools import CodeSearcher

# pytest summary timings such as " in 0.03s" or " in 65.40s (0:01:05)"
PYTEST_DURATION_RE = re.compile(r" in \d+(?:\.\d+)?s(?: \(\d+:\d{2}:\d{2}\))?")
# Traceback frames: pytest --tb=short ("parser.py:4: in first_field") and CPython ('File "x.py", line 4')
TRACE_FRAME_RE = re.compile(r'^(\S+\.py):\d+: in \S+|File "([^"]+\.py)", line \d+', re.MULTILINE)


def _is_test_file(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    name = parts[-1] if parts else ""
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py" or "tests" in parts[:-1]


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

    @staticmethod
    def failure_text(task: Task, test_result: TestResult) -> str:
        """
        Test output for prompts with per-run noise (timings, the throwaway workspace path) removed,
        so the same failure always produces the same prompt.
        """
        text = f"{test_result.stdout}\n{test_result.stderr}".replace(str(task.repository_path), ".")
        return PYTEST_DURATION_RE.sub("", text)

    @staticmethod
    def _repo_relative(repo: Path, raw_path: str) -> Optional[str]:
        """Map a path printed in test output to a file inside the repository, if it is one."""
        path = Path(raw_path)
        try:
            rel = path.resolve().relative_to(repo.resolve()) if path.is_absolute() else path
        except ValueError:
            return None  # outside the repository, e.g. site-packages
        return rel.as_posix() if (repo / rel).is_file() else None

    def locate_target_file(self, task: Task, test_result: TestResult, searcher: CodeSearcher) -> str:
        """
        Determine which source file contains the defect:
        1. an explicit metadata.target_file hint;
        2. the deepest non-test frame of the failing test's traceback;
        3. for failures raised in the test itself (assertions), the repository module the failing
           test imports, preferring one whose imported names appear in the failure output;
        4. the first non-test Python file.
        """
        if "target_file" in task.metadata:
            return task.metadata["target_file"]

        repo = task.repository_path
        output = f"{test_result.stdout}\n{test_result.stderr}"
        frames = [
            rel
            for match in TRACE_FRAME_RE.finditer(output)
            if (rel := self._repo_relative(repo, match.group(1) or match.group(2)))
        ]
        source_frames = [f for f in frames if not _is_test_file(f)]
        if source_frames:
            return source_frames[-1]

        test_files = [f for f in frames if _is_test_file(f)]
        targeted = self._repo_relative(repo, task.targeted_test.split()[0].split("::")[0]) if task.targeted_test else None
        candidates: List[tuple] = []
        for test_file in dict.fromkeys(test_files + ([targeted] if targeted else [])):
            try:
                tree = ast.parse((repo / test_file).read_text(encoding="utf-8"))
            except (OSError, SyntaxError, ValueError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    modules, names = [node.module], [alias.name for alias in node.names]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                    names = [m.split(".")[-1] for m in modules]
                else:
                    continue
                for module in modules:
                    for rel in (f"{module.replace('.', '/')}.py", f"{module.replace('.', '/')}/__init__.py"):
                        if (repo / rel).is_file() and not _is_test_file(rel):
                            mentioned = any(name in output for name in names)
                            candidates.append((not mentioned, rel))
        if candidates:
            return min(candidates, key=lambda c: c[0])[1]

        files = searcher.list_files(".py")
        non_test_files = [f for f in files if not _is_test_file(f)]
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
        temperature: Optional[float] = None,
    ) -> ModelPrompt:
        """Construct prompt incorporating strategy genes, context, retry feedback, and memory lessons."""
        signature = self.extract_failure_signature(pre_test)
        target_file = self.locate_target_file(task, pre_test, searcher)
        failure_trace = self.failure_text(task, pre_test)

        # 1. Assemble context respecting enable_multi_file_context gene
        content = ""
        if enable_multi_file_context:
            try:
                from evoagent.intelligence.context_builder import MultiFileContextBuilder
                builder = MultiFileContextBuilder(task.repository_path)
                ctx = builder.build_context(
                    primary_file=target_file,
                    failure_trace=failure_trace,
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
            failure_trace=failure_trace,
            max_results=retrieval_top_k,
            policy=retrieval_policy,
        )

        style_instruction = self._get_prompt_style_instruction(prompt_style)

        return ModelPrompt(
            task_id=task.task_id,
            issue_description=task.issue_description,
            failure_trace=failure_trace,
            target_file=target_file,
            file_content=content,
            retrieved_lessons=lessons,
            attempt_number=attempt,
            previous_attempt_patch=previous_attempt_patch,
            previous_attempt_error=previous_attempt_error,
            system_instruction=style_instruction,
            temperature=temperature,
        )
