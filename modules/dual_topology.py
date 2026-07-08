"""Public dual-topology contact summary.

This module keeps the key VTEC intuition without exposing the full private
controller: track two lightweight topological summaries and convert them into
safe control hints.

- hole topology: whether the peg is converging toward the insertion center.
- contact topology: whether the peg is getting locked around a contact obstacle.
- lock score: a bounded risk score used to reduce push force and bias release.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from modules.common import clip


@dataclass
class DualTopologyState:
    hole_phase: float = 0.0
    contact_phase: float = 0.0
    hole_turns: float = 0.0
    contact_turns: float = 0.0
    prev_hole_phase: float = 0.0
    prev_contact_phase: float = 0.0

    @staticmethod
    def _wrap_delta(delta: float) -> float:
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        return delta

    def update(
        self,
        lateral_error: float,
        force: float,
        tightness: float,
        observed_side: float,
        progress: float,
    ) -> Dict[str, float]:
        hole_phase = math.atan2(lateral_error, max(0.05, 1.0 - progress))
        contact_phase = math.atan2(observed_side * tightness, max(0.05, 1.2 - force / 110.0))

        dh = self._wrap_delta(hole_phase - self.prev_hole_phase)
        dc = self._wrap_delta(contact_phase - self.prev_contact_phase)
        self.hole_turns += dh / (2.0 * math.pi)
        self.contact_turns += dc / (2.0 * math.pi)
        self.prev_hole_phase = hole_phase
        self.prev_contact_phase = contact_phase
        self.hole_phase = hole_phase
        self.contact_phase = contact_phase

        hole_error = clip(abs(hole_phase) / math.pi, 0.0, 1.0)
        contact_lock = clip(abs(self.contact_turns) * 1.15 + tightness * 0.48 + clip(force / 130.0, 0.0, 1.0) * 0.30, 0.0, 1.0)
        recapture_readiness = clip(1.0 - 0.80 * hole_error - 0.28 * contact_lock, 0.0, 1.0)
        push_scale = clip(1.0 - 0.42 * contact_lock, 0.50, 1.0)
        release_bias = -1 if self.contact_turns > 0 else 1

        return {
            "hole_error": hole_error,
            "contact_lock": contact_lock,
            "recapture_readiness": recapture_readiness,
            "push_scale": push_scale,
            "release_bias": release_bias,
            "hole_turns": self.hole_turns,
            "contact_turns": self.contact_turns,
        }
