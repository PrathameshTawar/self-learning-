"""
State machine engine governing the autonomous repair lifecycle and learning loop.
"""

from pathlib import Path
import shutil
import tempfile
import time
import uuid
from typing import Optional
from evoagent.domain.models import (
    AttemptRecord,
    MemoryEpisodeRecord,
    PatchProposal,
    Task,
    TaskResult,
    TestResult,
)
from evoagent.domain.states import TaskState
from evoagent.evaluation.evaluator import IndependentEvaluator
from evoagent.memory.store import IMemoryStore
from evoagent.orchestration.planner import TaskPlanner
from evoagent.orchestration.repair_agent import RepairAgent
from evoagent.tools.git_tools import GitInspector
from evoagent.tools.patch_tools import PatchTool
from evoagent.tools.search_tools import CodeSearcher
from evoagent.tools.test_tools import TestRunnerTool


class TaskStateMachine:
    """
    Finite State Machine driving the EvoAgent task lifecycle:
    QUEUED -> ANALYZING -> PLANNED -> PATCHING -> EVALUATING -> SUCCEEDED/RETRYABLE_FAILURE -> LEARNING_REVIEW -> COMPLETED
    """

    def __init__(
        self,
        planner: TaskPlanner,
        repair_agent: RepairAgent,
        evaluator: IndependentEvaluator,
        memory_store: IMemoryStore,
        patch_tool: PatchTool,
        git_inspector: GitInspector,
        test_runner: TestRunnerTool,
    ) -> None:
        self.planner = planner
        self.repair_agent = repair_agent
        self.evaluator = evaluator
        self.memory_store = memory_store
        self.patch_tool = patch_tool
        self.git_inspector = git_inspector
        self.test_runner = test_runner

    def execute_task(self, task: Task) -> TaskResult:
        """
        Run the end-to-end autonomous repair loop on a task.
        Executes entirely within an isolated throwaway temporary workspace to protect
        the original repository and ensure idempotent evaluations.
        """
        start_time = time.perf_counter()
        attempts = []
        accepted_patch: Optional[PatchProposal] = None

        with tempfile.TemporaryDirectory(prefix=f"evo_{task.task_id}_") as temp_dir:
            temp_path = Path(temp_dir)
            clean_dir = temp_path / "clean_repo"
            work_dir = temp_path / "work_repo"

            # 1. Create pristine throwaway isolated copy of repository
            shutil.copytree(
                task.repository_path,
                clean_dir,
                ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"),
            )
            shutil.copytree(clean_dir, work_dir)

            isolated_task = task.model_copy(update={"repository_path": work_dir})
            searcher = CodeSearcher(work_dir)
            isolated_patch_tool = PatchTool(root_path=work_dir, max_patch_lines=self.patch_tool.max_patch_lines)

            # 2. Pre-patch verification gate - confirm bug reproduces initially
            pre_test: TestResult = self.evaluator.verify_reproduction(isolated_task)
            if pre_test.passed:
                # Targeted test already passes before patch - invalid task definition
                invalid_record = AttemptRecord(
                    attempt_number=1,
                    state=TaskState.INVALID_TASK,
                    pre_test_result=pre_test,
                    hypothesis="Task rejected: targeted test already passed pre-patch without any bug reproduction.",
                )
                attempts.append(invalid_record)
                total_duration = time.perf_counter() - start_time
                return TaskResult(
                    task_id=task.task_id,
                    final_state=TaskState.INVALID_TASK,
                    is_successful=False,
                    attempts=attempts,
                    accepted_patch=None,
                    total_duration_seconds=round(total_duration, 2),
                )

            failure_sig = self.planner.extract_failure_signature(pre_test)

            # 3. Main repair loop with feedback
            prev_attempt_patch: Optional[str] = None
            prev_attempt_error: Optional[str] = None

            for attempt_no in range(1, task.budget.max_attempts + 1):
                attempt_record = AttemptRecord(
                    attempt_number=attempt_no,
                    state=TaskState.ANALYZING,
                    pre_test_result=pre_test,
                )

                # Step 3a: Planning & memory retrieval (incorporates prior failure feedback on retries)
                prompt = self.planner.plan_repair(
                    task=isolated_task,
                    pre_test=pre_test,
                    searcher=searcher,
                    attempt=attempt_no,
                    previous_attempt_patch=prev_attempt_patch,
                    previous_attempt_error=prev_attempt_error,
                )
                attempt_record.state = TaskState.PLANNED

                # Step 3b: Patch generation
                candidate_patch = self.repair_agent.generate_candidate_patch(prompt)
                attempt_record.state = TaskState.PATCHING
                attempt_record.patch = candidate_patch
                # Store the model's actual rationale / hypothesis, not the task description
                attempt_record.hypothesis = candidate_patch.rationale or "Hypothesis formed by model backend."

                if not candidate_patch.diff_unified or not candidate_patch.is_valid_format:
                    prev_attempt_patch = candidate_patch.diff_unified or "(empty diff)"
                    prev_attempt_error = candidate_patch.validation_error or "Empty or malformed patch proposal."
                    attempt_record.state = TaskState.RETRYABLE_FAILURE
                    attempts.append(attempt_record)
                    continue

                # Step 3c: Apply patch inside throwaway workspace
                apply_success, apply_msg = isolated_patch_tool.apply_unified_diff(candidate_patch)
                if not apply_success:
                    prev_attempt_patch = candidate_patch.diff_unified
                    prev_attempt_error = f"Patch application failed: {apply_msg}"
                    attempt_record.state = TaskState.RETRYABLE_FAILURE
                    attempts.append(attempt_record)
                    # Restore clean working directory
                    shutil.rmtree(work_dir, ignore_errors=True)
                    shutil.copytree(clean_dir, work_dir)
                    continue

                # Step 3d: Independent Evaluation
                attempt_record.state = TaskState.EVALUATING
                post_test = self.test_runner.run_pytest(
                    test_path=isolated_task.targeted_test,
                    cwd=isolated_task.repository_path,
                    timeout_seconds=isolated_task.budget.timeout_seconds,
                )
                attempt_record.post_test_result = post_test

                evaluation = self.evaluator.evaluate_repair(isolated_task, candidate_patch, post_test)
                attempt_record.evaluation = evaluation

                if evaluation.is_accepted:
                    attempt_record.state = TaskState.SUCCEEDED
                    attempts.append(attempt_record)
                    accepted_patch = candidate_patch
                    break
                else:
                    # Capture failed patch and test error to feed into next attempt
                    prev_attempt_patch = candidate_patch.diff_unified
                    prev_attempt_error = evaluation.feedback or post_test.error_summary or post_test.stderr
                    # Clean reset before next attempt
                    shutil.rmtree(work_dir, ignore_errors=True)
                    shutil.copytree(clean_dir, work_dir)
                    attempt_record.state = TaskState.RETRYABLE_FAILURE
                    attempts.append(attempt_record)

            # Step 4: Learning review & memory persistence (Layer A)
            final_state = TaskState.COMPLETED if accepted_patch else TaskState.FAILED
            total_duration = time.perf_counter() - start_time

            # Persist episode into memory store
            last_attempt = attempts[-1] if attempts else None
            patch_diff = accepted_patch.diff_unified if accepted_patch else (last_attempt.patch.diff_unified if last_attempt and last_attempt.patch else "")
            outcome_str = "accepted" if accepted_patch else "rejected"

            lessons = (
                f"Successfully resolved with hypothesis: {accepted_patch.rationale}. Addressed failure signature: {failure_sig}"
                if accepted_patch
                else f"Failed to resolve bug after {len(attempts)} attempts. Last failure signature: {failure_sig}"
            )

            episode = MemoryEpisodeRecord(
                episode_id=str(uuid.uuid4()),
                task_id=task.task_id,
                issue_description=task.issue_description,
                failure_signature=failure_sig,
                target_file=accepted_patch.target_file if accepted_patch else (last_attempt.patch.target_file if last_attempt and last_attempt.patch else ""),
                root_cause_hypothesis=accepted_patch.rationale if accepted_patch else (last_attempt.hypothesis if last_attempt else ""),
                applied_patch=patch_diff,
                outcome=outcome_str,
                targeted_tests_passed=bool(accepted_patch),
                regression_tests_passed=bool(accepted_patch and last_attempt and last_attempt.evaluation and last_attempt.evaluation.regression_tests_passed),
                lessons_learned=lessons,
            )
            self.memory_store.save_episode(episode)

            return TaskResult(
                task_id=task.task_id,
                final_state=final_state,
                is_successful=bool(accepted_patch),
                attempts=attempts,
                accepted_patch=accepted_patch,
                total_duration_seconds=round(total_duration, 2),
            )
