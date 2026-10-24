"""
Cross-file symbol graph and dependency resolver.
"""

from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel
from evoagent.intelligence.ast_symbols import CodeSymbol, SymbolExtractor, SymbolType
from evoagent.tools.search_tools import CodeSearcher


class SymbolReference(BaseModel):
    """Link between a calling symbol and its cross-file definition."""

    source_file: str
    target_file: str
    symbol_name: str
    target_symbol: CodeSymbol


class SymbolGraph:
    """
    Constructs an index of symbols across all repository files and resolves
    cross-file imports, definitions, and call relationships.
    """

    def __init__(self, root_path: Path) -> None:
        self.root_path = root_path
        self.extractor = SymbolExtractor(root_path)
        self.searcher = CodeSearcher(root_path)
        self.symbols_by_file: Dict[str, List[CodeSymbol]] = {}
        self.symbols_by_name: Dict[str, List[CodeSymbol]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Scan and index all python source files."""
        python_files = self.searcher.list_files(".py")
        for rel_path in python_files:
            file_symbols = self.extractor.extract_file_symbols(rel_path)
            self.symbols_by_file[rel_path] = file_symbols

            for sym in file_symbols:
                if sym.symbol_type != SymbolType.IMPORT:
                    self.symbols_by_name.setdefault(sym.name, []).append(sym)

    def find_symbol_definitions(self, name: str) -> List[CodeSymbol]:
        """Find definitions of a symbol name across the codebase."""
        return self.symbols_by_name.get(name, [])

    def resolve_reference(self, symbol_name: str, calling_file: str) -> Optional[CodeSymbol]:
        """
        Resolve a symbol referenced in calling_file to its target definition in another file.
        Inspects imported modules and cross-file references.
        """
        file_symbols = self.symbols_by_file.get(calling_file, [])
        imports = [s for s in file_symbols if s.symbol_type == SymbolType.IMPORT]

        # 1. Direct from-import match: from module import symbol_name
        for imp in imports:
            if imp.name == symbol_name and imp.imported_from:
                target_mod = imp.imported_from.replace(".", "/")
                candidates = [f"{target_mod}.py", f"src/{target_mod}.py"]
                for cand in candidates:
                    for sym in self.symbols_by_file.get(cand, []):
                        if sym.name == symbol_name:
                            return sym

        # 2. General symbol definition lookup across repository
        candidates = self.symbols_by_name.get(symbol_name, [])
        for sym in candidates:
            if sym.file_path != calling_file:
                return sym

        return None

    def get_cross_file_dependencies(self, file_path: str) -> List[CodeSymbol]:
        """
        Identify all external symbols in other files that are called from file_path.
        """
        symbols = self.symbols_by_file.get(file_path, [])
        called_names = set()
        for s in symbols:
            called_names.update(s.calls)

        cross_deps: List[CodeSymbol] = []
        for name in called_names:
            resolved = self.resolve_reference(name, file_path)
            if resolved and resolved not in cross_deps:
                cross_deps.append(resolved)

        return cross_deps
