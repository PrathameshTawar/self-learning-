"""
Unit tests for AST SymbolExtractor.
"""

from pathlib import Path
from evoagent.intelligence.ast_symbols import SymbolExtractor, SymbolType


def test_extract_symbols_from_python_file(tmp_path: Path):
    code_file = tmp_path / "sample.py"
    code_file.write_text(
        "import math\n"
        "from os import path\n\n"
        "def compute_total(x: int) -> int:\n"
        "    \"\"\"Compute sum.\"\"\"\n"
        "    return math.sqrt(x)\n\n"
        "class Calculator:\n"
        "    def add(self, a, b):\n"
        "        return a + b\n",
        encoding="utf-8",
    )

    extractor = SymbolExtractor(tmp_path)
    symbols = extractor.extract_file_symbols("sample.py")

    names = [s.name for s in symbols]
    assert "math" in names
    assert "path" in names
    assert "compute_total" in names
    assert "Calculator" in names
    assert "Calculator.add" in names

    # Verify function properties
    func_sym = next(s for s in symbols if s.name == "compute_total")
    assert func_sym.symbol_type == SymbolType.FUNCTION
    assert func_sym.docstring == "Compute sum."
    assert "compute_total" in func_sym.signature
