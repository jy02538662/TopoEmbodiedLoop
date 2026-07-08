"""Simplified VTEC-style contact controller.

The full VTEC project has richer phase/winding control. This public demo keeps
only the intuitive behavior: insert, release, recapture, and retreat under a
safety guard and memory prior.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from modules.common import STATE_ID


@dataclass
class ContactController:
    def select_action(
        self,
        mode: str,
        belief: List[float],
        memory_prior: Dict[str, object],
        rng: random.Random,
    ) -> Tuple[str, int]:
        bias = int(memory_prior.get("release_bias", 0))
        if mode == "retreat":
            return "retreat", 0
        if mode == "release":
            return "release", bias if bias else rng.choice([-1, 1])
        if mode in {"slow_down", "reobserve"} and belief[STATE_ID["jam"]] > 0.30:
            return "release", bias if bias else rng.choice([-1, 1])
        if belief[STATE_ID["release"]] > 0.24 or belief[STATE_ID["recapture"]] > 0.22:
            return "recapture", 0
        if belief[STATE_ID["jam"]] > 0.38 or belief[STATE_ID["slip"]] > 0.42:
            return "release", bias if bias else rng.choice([-1, 1])
        return "insert", 0

    def reactive_action(self, probs: List[float], rng: random.Random) -> Tuple[str, int]:
        if probs[STATE_ID["jam"]] > 0.48:
            return "release", rng.choice([-1, 1])
        if probs[STATE_ID["release"]] > 0.45:
            return "recapture", 0
        return "insert", 0
