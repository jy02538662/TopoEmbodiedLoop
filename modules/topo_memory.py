"""Simplified TideMemory-style episodic adapter.

The full TideMemory prototype stores information in topological vortex winding
numbers. This open-source demo keeps the control-facing idea: recall similar
contact episodes and return actionable priors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from modules.common import STATE_ID


def distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass
class TopoMemory:
    k: int = 5
    items: List[Dict[str, object]] = field(default_factory=list)

    def signature(self, scene_id: int, belief: List[float], force: float, tightness: float, progress: float) -> List[float]:
        return [
            scene_id / 2.0,
            belief[STATE_ID["jam"]],
            belief[STATE_ID["slip"]],
            belief[STATE_ID["release"]],
            min(1.0, force / 130.0),
            tightness,
            progress,
        ]

    def recall(self, sig: List[float]) -> Dict[str, object]:
        if not self.items:
            return {"risk_prior": 0.0, "release_bias": 0, "confidence": 0.0}
        ranked = sorted(((distance(sig, item["signature"]), item) for item in self.items), key=lambda x: x[0])
        nearest = [item for d, item in ranked[: self.k] if d < 0.85]
        if not nearest:
            return {"risk_prior": 0.0, "release_bias": 0, "confidence": 0.0}
        risk = sum(not item["success"] for item in nearest) / len(nearest)
        successful_dirs = [int(item["release_dir"]) for item in nearest if item["success"] and int(item["release_dir"]) != 0]
        if successful_dirs:
            release_bias = 1 if sum(d > 0 for d in successful_dirs) >= sum(d < 0 for d in successful_dirs) else -1
        else:
            release_bias = 0
        return {
            "risk_prior": risk,
            "release_bias": release_bias,
            "confidence": min(1.0, len(nearest) / self.k),
        }

    def store(self, sig: List[float], success: bool, release_dir: int) -> None:
        self.items.append({"signature": list(sig), "success": bool(success), "release_dir": int(release_dir)})
