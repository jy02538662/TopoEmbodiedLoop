"""Simplified VTEC-style contact controller.

This public controller keeps the visible VTEC intuition while avoiding private
implementation details: a dual-topology hint modulates push strength and biases
release/recapture decisions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from modules.common import STATE_ID, clip


@dataclass
class ContactController:
    def select_action(
        self,
        mode: str,
        belief: List[float],
        memory_prior: Dict[str, object],
        topo_hint: Dict[str, float],
        rng: random.Random,
    ) -> Tuple[str, int, float]:
        memory_bias = int(memory_prior.get("release_bias", 0))
        topo_bias = int(topo_hint.get("release_bias", 0))
        release_dir = memory_bias if memory_bias else topo_bias if topo_bias else rng.choice([-1, 1])
        contact_lock = float(topo_hint.get("contact_lock", 0.0))
        push_scale = float(topo_hint.get("push_scale", 1.0))

        if mode == "retreat":
            return "retreat", 0, 0.45
        if mode == "release" or contact_lock > 0.84:
            return "release", release_dir, 0.60
        if mode in {"slow_down", "reobserve"} and belief[STATE_ID["jam"]] > 0.30:
            return "release", release_dir, 0.58
        if float(topo_hint.get("recapture_readiness", 0.0)) > 0.38 and belief[STATE_ID["release"]] > 0.14:
            return "recapture", 0, clip(push_scale, 0.55, 0.98)
        if belief[STATE_ID["release"]] > 0.20 or belief[STATE_ID["recapture"]] > 0.20:
            return "recapture", 0, clip(push_scale, 0.55, 0.98)
        if belief[STATE_ID["jam"]] > 0.34 or belief[STATE_ID["slip"]] > 0.44:
            return "release", release_dir, 0.58
        return "insert", 0, push_scale

    def reactive_action(self, probs: List[float], rng: random.Random) -> Tuple[str, int, float]:
        if probs[STATE_ID["jam"]] > 0.48:
            return "release", rng.choice([-1, 1]), 1.0
        if probs[STATE_ID["release"]] > 0.45:
            return "recapture", 0, 1.0
        return "insert", 0, 1.0
