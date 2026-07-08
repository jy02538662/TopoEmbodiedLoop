"""Simplified TopoWave + TopoReasoner.

ContactPerception mimics TopoWave: it maps sensor features to contact-state
probabilities. TopoReasoner then filters those probabilities through a topology-
aware transition model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from modules.common import BAD_STATE_IDS, STATE_ID, STATE_NAMES, argmax, clip, normalize


@dataclass
class ContactPerception:
    noise_floor: float = 0.06

    def predict(self, force: float, force_slope: float, vibration: float, optical_lock: float) -> Dict[str, List[float]]:
        base = [self.noise_floor] * len(STATE_NAMES)
        if force < 18 and vibration < 0.25:
            base[STATE_ID["free"]] += 0.70
        elif force < 45 and optical_lock < 0.45:
            base[STATE_ID["contact"]] += 0.70
        elif vibration > 0.55 and force < 82:
            base[STATE_ID["slip"]] += 0.72
        elif force > 78 or force_slope > 35:
            base[STATE_ID["jam"]] += 0.76
        elif force_slope < -25:
            base[STATE_ID["release"]] += 0.70
        else:
            base[STATE_ID["recapture"]] += 0.58
        probs = normalize(base)
        return {"state_probs": probs, "state": STATE_NAMES[argmax(probs)]}


@dataclass
class TopoReasoner:
    inertia: float = 0.72
    belief: List[float] = field(default_factory=lambda: [1.0 / len(STATE_NAMES)] * len(STATE_NAMES))
    transition: List[List[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.transition:
            self.transition = self._build_transition()

    def _build_transition(self) -> List[List[float]]:
        n = len(STATE_NAMES)
        mat = [[0.015 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            mat[i][i] += 0.58
        edges = [
            ("free", "contact", 0.22),
            ("contact", "slip", 0.18),
            ("slip", "jam", 0.24),
            ("jam", "release", 0.32),
            ("release", "recapture", 0.32),
            ("recapture", "contact", 0.18),
            ("recapture", "free", 0.14),
            ("slip", "release", 0.10),
        ]
        for src, dst, weight in edges:
            mat[STATE_ID[src]][STATE_ID[dst]] += weight
        return [normalize(row) for row in mat]

    def update(self, probs: List[float], reliability: float) -> Dict[str, object]:
        reliability = clip(reliability, 0.05, 1.0)
        uniform = [1.0 / len(probs)] * len(probs)
        likelihood = [reliability * p + (1.0 - reliability) * u for p, u in zip(probs, uniform)]
        prior = []
        for dst in range(len(probs)):
            prior.append(sum(self.transition[src][dst] * self.belief[src] for src in range(len(probs))))
        posterior = normalize([l * p for l, p in zip(likelihood, prior)])
        self.belief = normalize([
            self.inertia * post + (1.0 - self.inertia) * like
            for post, like in zip(posterior, likelihood)
        ])
        entropy = -sum(b * math.log(b + 1e-12) for b in self.belief) / math.log(len(self.belief))
        state_id = argmax(self.belief)
        return {
            "belief": list(self.belief),
            "state": STATE_NAMES[state_id],
            "uncertainty": entropy,
            "jam_prob": self.belief[STATE_ID["jam"]],
            "bad_prob": sum(self.belief[i] for i in BAD_STATE_IDS),
        }
