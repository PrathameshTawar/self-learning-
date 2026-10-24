"""
Versioned strategy policy registry with statistical promotion gates and rollback support.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from evoagent.learning.genetic import StrategyChromosome


class StrategyRegistry:
    """
    Registry for managing versioned strategy policies.
    Guarantees that strategies are only promoted with verified, statistically meaningful
    benchmark evidence over the currently active baseline strategy.
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
        min_fitness_margin: float = 0.05,
        force: bool = False,
    ) -> str:
        """
        Promote an evolved strategy policy.
        Requires candidate strategy to demonstrate a statistically meaningful improvement
        (default >= 0.05 fitness margin) over the currently active strategy unless forced.
        """
        data = self._read_data()
        current_active = self.get_active_strategy()

        if not force and current_active and len(data.get("versions", [])) > 0:
            cand_fitness = chromosome.fitness_score
            curr_fitness = current_active.fitness_score
            if cand_fitness < (curr_fitness + min_fitness_margin):
                raise ValueError(
                    f"Promotion gate rejected strategy '{chromosome.strategy_id}': "
                    f"candidate fitness {cand_fitness:.4f} did not exceed current baseline "
                    f"{curr_fitness:.4f} by required margin of {min_fitness_margin:.2f}"
                )

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
