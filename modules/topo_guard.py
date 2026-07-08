"""Risk-aware emergency guard for contact-rich control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from modules.common import STATE_ID, clip


@dataclass
class TopoGuard:
    force_budget: float = 105.0
    hard_force: float = 132.0
    impulse_limit: float = 70.0
    dt: float = 0.05
    impulse: float = 0.0
    prev_force: float = 0.0

    def update(
        self,
        belief: List[float],
        force: float,
        tightness: float,
        uncertainty: float,
        reliability: float,
        memory_prior: Dict[str, object],
    ) -> Dict[str, object]:
        self.impulse += force * self.dt
        force_slope = (force - self.prev_force) / max(self.dt, 1e-9)
        self.prev_force = force
        risk = (
            0.34 * belief[STATE_ID["jam"]]
            + 0.18 * clip(force / self.force_budget, 0.0, 1.0)
            + 0.14 * clip(self.impulse / self.impulse_limit, 0.0, 1.0)
            + 0.16 * tightness
            + 0.10 * uncertainty
            + 0.05 * (1.0 - reliability)
            + 0.03 * float(memory_prior.get("risk_prior", 0.0))
        )
        if force > self.hard_force or risk > 0.88:
            mode = "retreat"
        elif belief[STATE_ID["jam"]] > 0.45 and force_slope > 30:
            mode = "release"
        elif tightness > 0.78 or risk > 0.66:
            mode = "slow_down"
        elif uncertainty > 0.72 and reliability < 0.50:
            mode = "reobserve"
        else:
            mode = "continue"
        return {
            "mode": mode,
            "risk": risk,
            "impulse": self.impulse,
            "force_slope": force_slope,
            "scale": 0.62 if mode in {"slow_down", "reobserve"} else 1.0,
        }
