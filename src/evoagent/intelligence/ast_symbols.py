"""
AST symbol extraction parsing functions, classes, methods, and import references.
"""

import ast
from enum import Enum
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class SymbolType(str, Enum):
    """Types of code symbols identified in AST analysis."""

    FUNCTION = "FUNCTION"
    METHOD = "METHOD"
    CLASS = "CLASS"
    IMPORT = "IMPORT"
    VARIABLE = "VARIABLE"


class CodeSymbol(BaseModel):
    """Representation of an extracted code symbol with file coordinates and signature."""

    name: str
    symbol_type: SymbolType
    file_path: str
    start_line: int
    end_line: int
    signature: str = ""
    docstring: Optional[str] = None
    calls: List[str] = Field(default_factory=list)
    imported_from: Optional[str] = None


class SymbolExtractor:
    """Extracts symbols and syntax definitions from Python source code via AST."""

    def __init__(self, root_path: Path) -> None:
        self.root_path = root_path

    def extract_file_symbols(self, rel_path: str) -> List[CodeSymbol]:
        """Parse source file into structured CodeSymbol instances."""
        file_path = (self.root_path / rel_path).resolve()
        if not file_path.exists() or not file_path.is_file():
            return []

        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            return []

        symbols: List[CodeSymbol] = []
        lines = source.splitlines()

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(self._parse_function(node, rel_path, lines))
            elif isinstance(node, ast.ClassDef):
                symbols.extend(self._parse_class(node, rel_path, lines))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    symbols.append(
                        CodeSymbol(
                            name=alias.name,
                            symbol_type=SymbolType.IMPORT,
                            file_path=rel_path,
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            signature=f"import {alias.name}",
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    symbols.append(
                        CodeSymbol(
                            name=alias.name,
                            symbol_type=SymbolType.IMPORT,
                            file_path=rel_path,
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            signature=f"from {mod} import {alias.name}",
                            imported_from=mod,
                        )
                    )

        return symbols

    def _parse_function(
        self,
        node: ast.FunctionDef,
        rel_path: str,
        lines: List[str],
        parent_class: Optional[str] = None,
    ) -> CodeSymbol:
        """Parse function or method node."""
        calls = [n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        start = node.lineno
        end = node.end_lineno or start
        sig_line = lines[start - 1].strip() if start <= len(lines) else f"def {node.name}()"

        docstring = ast.get_docstring(node)
        sym_type = SymbolType.METHOD if parent_class else SymbolType.FUNCTION
        full_name = f"{parent_class}.{node.name}" if parent_class else node.name

        return CodeSymbol(
            name=full_name,
            symbol_type=sym_type,
            file_path=rel_path,
            start_line=start,
            end_line=end,
            signature=sig_line,
            docstring=docstring,
            calls=calls,
        )

    def _parse_class(
        self,
        node: ast.ClassDef,
        rel_path: str,
        lines: List[str],
    ) -> List[CodeSymbol]:
        """Parse class definition and its member methods."""
        symbols: List[CodeSymbol] = []
        start = node.lineno
        end = node.end_lineno or start
        sig_line = lines[start - 1].strip() if start <= len(lines) else f"class {node.name}"

        symbols.append(
            CodeSymbol(
                name=node.name,
                symbol_type=SymbolType.CLASS,
                file_path=rel_path,
                start_line=start,
                end_line=end,
                signature=sig_line,
                docstring=ast.get_docstring(node),
            )
        )

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(self._parse_function(item, rel_path, lines, parent_class=node.name))

        return symbols
