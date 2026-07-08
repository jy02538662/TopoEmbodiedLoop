"""Simplified DualWave-style reliability gate.

The real DualWave project estimates which modality should be trusted. This
open-source demo keeps the idea compact: compare current sensor quality with a
short warmup baseline and output a reliability score in [0, 1].
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Deque, Dict, List

from modules.common import clip


@dataclass
class ReliabilityGate:
    window: int = 12
    min_reliability: float = 0.15
    audio_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=12))
    optical_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=12))

    def _score(self, value: float, hist: Deque[float]) -> float:
        if len(hist) < 4:
            hist.append(value)
            return 0.85
        mu = mean(hist)
        sigma = pstdev(hist) + 1e-6
        z = abs(value - mu) / sigma
        score = 1.0 / (1.0 + 0.35 * z)
        hist.append(value)
        return clip(score, self.min_reliability, 1.0)

    def update(self, audio_quality: float, optical_quality: float) -> Dict[str, float]:
        audio = self._score(audio_quality, self.audio_hist)
        optical = self._score(optical_quality, self.optical_hist)
        combined = 0.55 * audio + 0.45 * optical
        return {
            "audio": audio,
            "optical": optical,
            "combined": clip(combined, self.min_reliability, 1.0),
            "uncertainty": 1.0 - clip(combined, self.min_reliability, 1.0),
        }

    def fuse(self, audio_probs: List[float], optical_probs: List[float], reliability: Dict[str, float]) -> List[float]:
        a = reliability["audio"]
        o = reliability["optical"]
        denom = a + o + 1e-12
        return [(a * pa + o * po) / denom for pa, po in zip(audio_probs, optical_probs)]
