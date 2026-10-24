"""
End-to-end test for the strategy-evolve CLI command on the bundled benchmark split.
"""

from pathlib import Path
from click.testing import CliRunner
from evoagent.cli import main

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


def test_strategy_evolve_scores_on_held_out_and_survives_gate_rejection(tmp_path: Path, monkeypatch):
    # The command writes data/ relative to the working directory
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    args = ["strategy-evolve", "--generations", "1", "--population", "2", "--benchmarks", str(BENCHMARKS)]

    first = runner.invoke(main, args)
    assert first.exit_code == 0, first.output
    assert "held-out fitness" in first.output
    assert "promoted as version v1" in first.output

    # Same champion quality again: the promotion gate rejects it without crashing the command
    second = runner.invoke(main, args)
    assert second.exit_code == 0, second.output
    assert "Champion not promoted" in second.output
