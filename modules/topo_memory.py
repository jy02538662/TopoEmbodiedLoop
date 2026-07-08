"""Simplified TideMemory-style episodic adapter.

The full TideMemory prototype stores information in topological vortex winding
numbers. This open-source demo keeps the control-facing idea, but adds a public
"topological tag" layer so it is not just a plain nearest-neighbor memory:

- topo_tag: a coarse symbolic winding-like label for the dominant contact mode.
- phase_bin: a coarse phase bucket derived from the current episode signature.
- recall: combine geometric proximity with topological tag agreement.

This keeps the module lightweight while preserving a visible topological flavor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from modules.common import STATE_ID, argmax, clip


def distance(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _phase_bucket(value: float, bins: int = 8) -> int:
    value = clip(value, 0.0, 0.999999)
    return min(bins - 1, int(value * bins))


@dataclass
class TopoMemory:
    k: int = 5
    items: List[Dict[str, object]] = field(default_factory=list)

    def _topo_tag(self, belief: List[float], force: float, tightness: float, progress: float) -> int:
        dominant = argmax(belief)
        jam = belief[STATE_ID["jam"]]
        slip = belief[STATE_ID["slip"]]
        release = belief[STATE_ID["release"]]
        recapture = belief[STATE_ID["recapture"]]
        if jam >= 0.33 or (force > 92.0 and tightness > 0.72):
            return 2
        if slip >= 0.28 or tightness > 0.64:
            return 1
        if release >= 0.24:
            return -1
        if recapture >= 0.22 or progress > 0.70:
            return -2
        if dominant == STATE_ID["contact"]:
            return 0
        return 0

    def signature(
        self,
        scene_id: int,
        belief: List[float],
        force: float,
        tightness: float,
        progress: float,
        topo_hint: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        topo_tag = self._topo_tag(belief, force, tightness, progress)
        if topo_hint is None:
            topo_hint = {}
        contact_lock = float(topo_hint.get("contact_lock", 0.0))
        hole_error = float(topo_hint.get("hole_error", 0.0))
        recapture_readiness = float(topo_hint.get("recapture_readiness", 0.0))
        push_scale = float(topo_hint.get("push_scale", 1.0))
        phase = clip(
            0.24 * belief[STATE_ID["jam"]]
            + 0.18 * belief[STATE_ID["slip"]]
            + 0.12 * belief[STATE_ID["release"]]
            + 0.12 * contact_lock
            + 0.10 * hole_error
            + 0.10 * (1.0 - recapture_readiness)
            + 0.08 * (1.0 - push_scale)
            + 0.08 * clip(force / 130.0, 0.0, 1.0),
            0.0,
            0.999999,
        )
        return [
            scene_id / 2.0,
            topo_tag / 2.0,
            _phase_bucket(phase) / 8.0,
            belief[STATE_ID["jam"]],
            belief[STATE_ID["slip"]],
            belief[STATE_ID["release"]],
            min(1.0, force / 130.0),
            clip(tightness, 0.0, 1.0),
            clip(progress, 0.0, 1.0),
            clip(contact_lock, 0.0, 1.0),
            clip(hole_error, 0.0, 1.0),
            clip(recapture_readiness, 0.0, 1.0),
        ]

    def recall(self, sig: List[float]) -> Dict[str, object]:
        if not self.items:
            return {"risk_prior": 0.0, "release_bias": 0, "confidence": 0.0, "topo_match": 0.0}

        scored = []
        sig_tag = round(sig[1] * 2.0)
        sig_phase = sig[2]
        for item in self.items:
            topo_gap = abs(sig_tag - int(item.get("topo_tag", 0)))
            phase_gap = abs(sig_phase - float(item.get("phase_bin", 0) / 8.0))
            d = distance(sig, item["signature"])
            score = d + 0.18 * topo_gap + 0.08 * phase_gap
            scored.append((score, item))

        scored.sort(key=lambda x: x[0])
        nearest = [item for score, item in scored[: self.k] if score < 1.1]
        if not nearest:
            return {"risk_prior": 0.0, "release_bias": 0, "confidence": 0.0, "topo_match": 0.0}

        risk = sum(not item["success"] for item in nearest) / len(nearest)
        successful_dirs = [int(item["release_dir"]) for item in nearest if item["success"] and int(item["release_dir"]) != 0]
        failed_dirs = [int(item["release_dir"]) for item in nearest if not item["success"] and int(item["release_dir"]) != 0]
        if successful_dirs:
            release_bias = 1 if sum(d > 0 for d in successful_dirs) >= sum(d < 0 for d in successful_dirs) else -1
        elif failed_dirs:
            failed_majority = 1 if sum(d > 0 for d in failed_dirs) >= sum(d < 0 for d in failed_dirs) else -1
            release_bias = -failed_majority
        else:
            release_bias = 0

        topo_match = sum(1 for item in nearest if int(item.get("topo_tag", 0)) == sig_tag) / len(nearest)
        return {
            "risk_prior": risk,
            "release_bias": release_bias,
            "confidence": min(1.0, len(nearest) / self.k),
            "topo_match": topo_match,
        }

    def store(self, sig: List[float], success: bool, release_dir: int) -> None:
        self.items.append(
            {
                "signature": list(sig),
                "success": bool(success),
                "release_dir": int(release_dir),
                "topo_tag": int(round(sig[1] * 2.0)),
                "phase_bin": _phase_bucket(sig[2]),
            }
        )
