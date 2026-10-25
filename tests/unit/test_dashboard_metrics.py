"""
Unit tests for dashboard metric aggregation over stored episodes.
"""

from evoagent.dashboard.server import episode_metrics
from evoagent.domain.models import MemoryEpisodeRecord


def _episode(outcome: str, metadata: dict) -> MemoryEpisodeRecord:
    return MemoryEpisodeRecord(
        episode_id=f"ep-{outcome}-{len(metadata)}",
        task_id="t",
        issue_description="",
        failure_signature="",
        target_file="mod.py",
        root_cause_hypothesis="",
        applied_patch="",
        outcome=outcome,
        targeted_tests_passed=outcome == "accepted",
        regression_tests_passed=outcome == "accepted",
        lessons_learned="",
        metadata=metadata,
    )


def test_metrics_are_unknown_without_episodes():
    assert episode_metrics([]) == {
        "total_tasks": 0,
        "resolution_rate_percent": None,
        "regression_free_rate_percent": None,
        "patch_validity_rate_percent": None,
        "average_duration_seconds": None,
    }


def test_metrics_are_computed_from_episode_telemetry():
    telemetry_ok = {
        "duration_seconds": 2.0, "attempts": 2, "patches_generated": 2, "valid_patches": 2,
        "targeted_pass_candidates": 2, "regression_free_candidates": 1,
    }
    telemetry_failed = {
        "duration_seconds": 4.0, "attempts": 3, "patches_generated": 2, "valid_patches": 0,
        "targeted_pass_candidates": 0, "regression_free_candidates": 0,
    }
    metrics = episode_metrics([
        _episode("accepted", telemetry_ok),
        _episode("rejected", telemetry_failed),
        _episode("rejected", {}),  # older episode without telemetry
    ])

    assert metrics["total_tasks"] == 3
    assert metrics["resolution_rate_percent"] == 33.33
    assert metrics["regression_free_rate_percent"] == 50.0
    assert metrics["patch_validity_rate_percent"] == 50.0
    assert metrics["average_duration_seconds"] == 3.0
