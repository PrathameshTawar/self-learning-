"""
Multi-file context aggregator assembling symbol snippets and cross-file references.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from evoagent.intelligence.ast_symbols import CodeSymbol
from evoagent.intelligence.symbol_graph import SymbolGraph
from evoagent.tools.search_tools import CodeSearcher


class MultiFileContextBuilder:
    """
    Builds structured, multi-file code context by tracing cross-file symbol definitions
    and error callstacks within token constraints.
    """

    def __init__(self, root_path: Path, max_total_chars: int = 12000) -> None:
        self.root_path = root_path
        self.max_total_chars = max_total_chars
        self.searcher = CodeSearcher(root_path)
        self.graph = SymbolGraph(root_path)

    def build_context(
        self,
        primary_file: str,
        failure_trace: str = "",
        max_cross_file_symbols: int = 4,
    ) -> Dict[str, Any]:
        """
        Assemble primary file source and relevant cross-file dependency snippets.
        """
        # Read primary file
        try:
            primary_content = self.searcher.read_file(primary_file)
        except Exception:
            primary_content = ""

        # Identify cross-file dependencies
        cross_symbols = self.graph.get_cross_file_dependencies(primary_file)

        # Also inspect failure trace for mentioned files/symbols
        for name, sym_list in self.graph.symbols_by_name.items():
            if name in failure_trace:
                for s in sym_list:
                    if s.file_path != primary_file and s not in cross_symbols:
                        cross_symbols.append(s)

        referenced_snippets: List[Dict[str, Any]] = []
        accumulated_chars = len(primary_content)

        for sym in cross_symbols[:max_cross_file_symbols]:
            if accumulated_chars >= self.max_total_chars:
                break

            try:
                snippet = self.searcher.read_file(
                    sym.file_path,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                )
                referenced_snippets.append({
                    "symbol_name": sym.name,
                    "file_path": sym.file_path,
                    "signature": sym.signature,
                    "lines": f"{sym.start_line}-{sym.end_line}",
                    "content": snippet,
                })
                accumulated_chars += len(snippet)
            except Exception:
                continue

        # Format unified prompt context
        formatted_context = self._format_prompt_block(primary_file, primary_content, referenced_snippets)

        return {
            "primary_file": primary_file,
            "primary_content": primary_content,
            "cross_file_references": referenced_snippets,
            "formatted_context": formatted_context,
            "total_files_referenced": 1 + len(set(s["file_path"] for s in referenced_snippets)),
        }

    def _format_prompt_block(
        self,
        primary_file: str,
        primary_content: str,
        cross_snippets: List[Dict[str, Any]],
    ) -> str:
        """Format clean markdown block of multi-file code context."""
        parts = [
            f"=== PRIMARY DEFECT FILE: {primary_file} ===\n{primary_content}\n"
        ]

        if cross_snippets:
            parts.append("=== REFERENCED CROSS-FILE DEFINITIONS ===")
            for item in cross_snippets:
                parts.append(
                    f"\n--- {item['file_path']} (Symbol: {item['symbol_name']}, Lines {item['lines']}) ---\n"
                    f"{item['content']}"
                )

        return "\n".join(parts)
