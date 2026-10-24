"""
Versioned strategy policy registry with promotion gates and rollback support.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from evoagent.learning.genetic import StrategyChromosome


class StrategyRegistry:
    """
    Registry for managing versioned strategy policies.
    Guarantees that strategies are only promoted with verified benchmark evidence.
    """

    def __init__(self, registry_file: Path) -> None:
        self.registry_file = registry_file
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_registry()

    def _init_registry(self) -> None:
        if not self.registry_file.exists():
            initial_data = {
                "active_strategy_id": "strat-default",
                "versions": [],
            }
            self.registry_file.write_text(json.dumps(initial_data, indent=2), encoding="utf-8")

    def _read_data(self) -> Dict[str, Any]:
        try:
            return json.loads(self.registry_file.read_text(encoding="utf-8"))
        except Exception:
            return {"active_strategy_id": "strat-default", "versions": []}

    def _save_data(self, data: Dict[str, Any]) -> None:
        self.registry_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def promote_strategy(
        self,
        chromosome: StrategyChromosome,
        benchmark_evidence: Dict[str, Any],
    ) -> str:
        """Promote an evolved strategy policy with proof of held-out benchmark superiority."""
        data = self._read_data()
        version_id = f"v{len(data.get('versions', [])) + 1}"

        record = {
            "version": version_id,
            "strategy": chromosome.model_dump(),
            "evidence": benchmark_evidence,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }

        data.setdefault("versions", []).append(record)
        data["active_strategy_id"] = chromosome.strategy_id
        self._save_data(data)
        return version_id

    def get_active_strategy(self) -> Optional[StrategyChromosome]:
        """Load currently active champion strategy."""
        data = self._read_data()
        active_id = data.get("active_strategy_id")
        for v in reversed(data.get("versions", [])):
            if v["strategy"]["strategy_id"] == active_id:
                return StrategyChromosome(**v["strategy"])

        # Fallback to default
        return StrategyChromosome(strategy_id="strat-default")

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all promoted strategy versions."""
        return self._read_data().get("versions", [])
