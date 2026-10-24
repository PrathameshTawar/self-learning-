"""
State machine engine governing the autonomous repair lifecycle and learning loop.
"""

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
        """Run the end-to-end autonomous repair loop on a task."""
        start_time = time.perf_counter()
        current_state = TaskState.QUEUED
        attempts = []
        searcher = CodeSearcher(task.repository_path)
        accepted_patch: Optional[PatchProposal] = None

        current_state = TaskState.ANALYZING

        # Step 1: Pre-patch verification - confirm bug reproduces
        pre_test: TestResult = self.evaluator.verify_reproduction(task)
        failure_sig = self.planner.extract_failure_signature(pre_test)

        # Main repair loop
        for attempt_no in range(1, task.budget.max_attempts + 1):
            attempt_record = AttemptRecord(
                attempt_number=attempt_no,
                state=TaskState.ANALYZING,
                pre_test_result=pre_test,
            )

            # Step 2: Planning & memory retrieval
            current_state = TaskState.PLANNED
            attempt_record.state = current_state
            prompt = self.planner.plan_repair(task, pre_test, searcher, attempt=attempt_no)
            attempt_record.hypothesis = prompt.issue_description

            # Step 3: Patch generation
            current_state = TaskState.PATCHING
            attempt_record.state = current_state
            candidate_patch = self.repair_agent.generate_candidate_patch(prompt)
            attempt_record.patch = candidate_patch

            if not candidate_patch.diff_unified or not candidate_patch.is_valid_format:
                # Retryable failure
                attempt_record.state = TaskState.RETRYABLE_FAILURE
                attempts.append(attempt_record)
                continue

            # Step 4: Apply patch inside repository
            apply_success, apply_msg = self.patch_tool.apply_unified_diff(candidate_patch)
            if not apply_success:
                attempt_record.state = TaskState.RETRYABLE_FAILURE
                attempts.append(attempt_record)
                continue

            # Step 5: Independent Evaluation
            current_state = TaskState.EVALUATING
            attempt_record.state = current_state
            post_test = self.test_runner.run_pytest(
                test_path=task.targeted_test,
                cwd=task.repository_path,
                timeout_seconds=task.budget.timeout_seconds,
            )
            attempt_record.post_test_result = post_test

            evaluation = self.evaluator.evaluate_repair(task, candidate_patch, post_test)
            attempt_record.evaluation = evaluation

            if evaluation.is_accepted:
                current_state = TaskState.SUCCEEDED
                attempt_record.state = current_state
                attempts.append(attempt_record)
                accepted_patch = candidate_patch
                break
            else:
                # Rollback changes before next attempt
                self.git_inspector.reset_working_tree(task.repository_path)
                attempt_record.state = TaskState.RETRYABLE_FAILURE
                attempts.append(attempt_record)

        # Step 6: Learning review & memory persistence (Layer A)
        final_state = TaskState.COMPLETED if accepted_patch else TaskState.FAILED
        total_duration = time.perf_counter() - start_time

        # Persist episode into memory store
        last_attempt = attempts[-1] if attempts else None
        patch_diff = accepted_patch.diff_unified if accepted_patch else (last_attempt.patch.diff_unified if last_attempt and last_attempt.patch else "")
        outcome_str = "accepted" if accepted_patch else "rejected"

        lessons = (
            f"Successfully resolved with hypothesis: {last_attempt.hypothesis}"
            if accepted_patch
            else f"Failed to resolve bug after {len(attempts)} attempts."
        )

        episode = MemoryEpisodeRecord(
            episode_id=str(uuid.uuid4()),
            task_id=task.task_id,
            issue_description=task.issue_description,
            failure_signature=failure_sig,
            target_file=accepted_patch.target_file if accepted_patch else (last_attempt.patch.target_file if last_attempt and last_attempt.patch else ""),
            root_cause_hypothesis=last_attempt.hypothesis if last_attempt and last_attempt.hypothesis else "",
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
