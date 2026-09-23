from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from cage_system.types import CandidatePose


@dataclass
class SelectionResult:
    valid: bool
    best: Optional[CandidatePose]
    reason: str
    ranked_candidate_ids: List[str]


class CandidateSelector:
    """Strict repeated-EST-mean selector; feasibility is only a hard gate."""

    def __init__(self, est_config):
        self.minimum = float(est_config["minimum_retention_ratio"])

    @staticmethod
    def _primary(candidate: CandidatePose) -> float:
        return float(candidate.metrics.get("est_occupancy_mean", -np.inf))

    def select(self, candidates: List[CandidatePose]) -> SelectionResult:
        valid = [candidate for candidate in candidates if candidate.feasible and np.isfinite(self._primary(candidate))]
        if not valid:
            return SelectionResult(False, None, "no_geometrically_and_dynamically_evaluated_candidate", [])
        maximum = max(self._primary(candidate) for candidate in valid)
        if maximum < self.minimum:
            return SelectionResult(False, None, "all_est_retention_below_threshold", [])
        # No IK margin, joint delta, temporal continuity, pose smoothness or arc
        # midpoint term is allowed to perturb the EST optimum. candidate_id is
        # only a deterministic exact-tie key.
        ranked = sorted(
            valid,
            key=lambda item: (-self._primary(item), item.candidate_id),
        )
        best = ranked[0]
        for rank, candidate in enumerate(ranked):
            candidate.metrics["selection_rank"] = float(rank)
            candidate.metrics["final_score"] = self._primary(candidate)
        return SelectionResult(True, best, "", [candidate.candidate_id for candidate in ranked])
