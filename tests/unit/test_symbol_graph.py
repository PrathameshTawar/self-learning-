"""
Unit tests for SymbolGraph cross-file resolution and MultiFileContextBuilder.
"""

from pathlib import Path
from evoagent.intelligence.context_builder import MultiFileContextBuilder
from evoagent.intelligence.symbol_graph import SymbolGraph


def test_symbol_graph_resolves_cross_file_reference(tmp_path: Path):
    helper = tmp_path / "helper.py"
    helper.write_text(
        "def format_token(token: str) -> str:\n"
        "    return token.strip().lower()\n",
        encoding="utf-8",
    )

    main_file = tmp_path / "main.py"
    main_file.write_text(
        "from helper import format_token\n\n"
        "def run_pipeline(raw):\n"
        "    return format_token(raw)\n",
        encoding="utf-8",
    )

    graph = SymbolGraph(tmp_path)
    resolved = graph.resolve_reference("format_token", "main.py")

    assert resolved is not None
    assert resolved.name == "format_token"
    assert resolved.file_path == "helper.py"
    assert "format_token" in resolved.signature

    # Verify context builder aggregates cross-file definitions
    builder = MultiFileContextBuilder(tmp_path)
    context = builder.build_context("main.py")

    assert context["primary_file"] == "main.py"
    assert len(context["cross_file_references"]) >= 1
    assert context["cross_file_references"][0]["file_path"] == "helper.py"
    assert "=== REFERENCED CROSS-FILE DEFINITIONS ===" in context["formatted_context"]
