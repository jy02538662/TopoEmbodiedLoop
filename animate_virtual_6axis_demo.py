"""Generate a no-hardware 3D/6-axis virtual contact demo.

This script is a visual mechanism demo, not a real-robot benchmark. It shows how
six-axis force/torque-like signals can be mapped into a compact topological
contact-lock estimate and a lateral release bias.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from modules.common import clip


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CSV_PATH = RESULTS / "virtual_6axis_demo.csv"


class DualTopologyState3D:
    def __init__(self) -> None:
        self.contact_turns_x = 0.0
        self.contact_turns_y = 0.0
        self.prev_contact_phase_x = 0.0
        self.prev_contact_phase_y = 0.0

    @staticmethod
    def _wrap_delta(delta: float) -> float:
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        return delta

    def update(
        self,
        err_x: float,
        err_y: float,
        fx: float,
        fy: float,
        fz: float,
        tx: float,
        ty: float,
        tz: float,
        progress: float,
    ) -> Dict[str, float]:
        del err_x, err_y, tz, progress
        denom_f = max(0.05, 1.2 - fz / 150.0)
        contact_phase_x = math.atan2(0.60 * fx - 0.35 * ty, denom_f)
        contact_phase_y = math.atan2(0.60 * fy + 0.35 * tx, denom_f)
        dc_x = self._wrap_delta(contact_phase_x - self.prev_contact_phase_x)
        dc_y = self._wrap_delta(contact_phase_y - self.prev_contact_phase_y)
        self.contact_turns_x += dc_x / (2.0 * math.pi)
        self.contact_turns_y += dc_y / (2.0 * math.pi)
        self.prev_contact_phase_x = contact_phase_x
        self.prev_contact_phase_y = contact_phase_y

        torque_level = clip((abs(tx) + abs(ty)) / 30.0, 0.0, 1.0)
        force_level = clip(fz / 150.0, 0.0, 1.0)
        turn_level = clip(math.sqrt(self.contact_turns_x**2 + self.contact_turns_y**2) * 1.25, 0.0, 1.0)
        contact_lock = clip(0.42 * turn_level + 0.34 * torque_level + 0.24 * force_level, 0.0, 1.0)
        push_scale = clip(1.0 - 0.52 * contact_lock, 0.34, 1.0)
        release_bias_x = -1.0 if self.contact_turns_x > 0 else 1.0
        release_bias_y = -1.0 if self.contact_turns_y > 0 else 1.0
        return {
            "contact_lock": contact_lock,
            "push_scale": push_scale,
            "release_bias_x": release_bias_x,
            "release_bias_y": release_bias_y,
            "turns_x": self.contact_turns_x,
            "turns_y": self.contact_turns_y,
        }


@dataclass
class Virtual6AxisAssemblyEnv:
    progress: float = 0.0
    pos_x: float = 0.15
    pos_y: float = -0.12
    failed: bool = False
    success: bool = False

    def observe(self, cmd_z: float) -> Dict[str, float]:
        err_x = self.pos_x
        err_y = self.pos_y
        dist = math.sqrt(err_x**2 + err_y**2)
        fx = 0.0
        fy = 0.0
        fz = 10.0 + 4.0 * cmd_z
        tx = 0.0
        ty = 0.0
        tz = 0.0
        if 0.37 < self.progress < 0.68 and dist > 0.035:
            fz = 10.0 + cmd_z * 165.0 * clip(dist * 3.2, 0.0, 1.0)
            fx = (err_x / dist) * fz * 0.36
            fy = (err_y / dist) * fz * 0.36
            tx = fy * 1.25
            ty = -fx * 1.25
            tz = (fx + fy) * 0.16
        return {"err_x": err_x, "err_y": err_y, "fx": fx, "fy": fy, "fz": fz, "tx": tx, "ty": ty, "tz": tz}

    def step(self, cmd_z: float, cmd_x: float, cmd_y: float) -> Dict[str, float]:
        cmd_z = clip(cmd_z, 0.0, 1.0)
        cmd_x = clip(cmd_x, -1.0, 1.0)
        cmd_y = clip(cmd_y, -1.0, 1.0)
        obs = self.observe(cmd_z)
        dist = math.sqrt(obs["err_x"] ** 2 + obs["err_y"] ** 2)

        if 0.37 < self.progress < 0.68 and dist > 0.035:
            drift_x = (obs["err_x"] / dist) * 0.012 * cmd_z
            drift_y = (obs["err_y"] / dist) * 0.012 * cmd_z
            release_x = cmd_x * 0.028
            release_y = cmd_y * 0.028
            self.pos_x += drift_x + release_x
            self.pos_y += drift_y + release_y
            self.progress += max(0.0, 0.014 * cmd_z * (1.0 - clip(dist * 3.0, 0.0, 0.9)))
        else:
            self.pos_x += cmd_x * 0.014 - self.pos_x * 0.050
            self.pos_y += cmd_y * 0.014 - self.pos_y * 0.050
            self.progress += 0.048 * cmd_z

        if self.progress >= 0.68 and math.sqrt(self.pos_x**2 + self.pos_y**2) < 0.075:
            self.pos_x *= 0.62
            self.pos_y *= 0.62
            self.progress += 0.038 * cmd_z

        self.progress = clip(self.progress, 0.0, 1.0)
        obs = self.observe(cmd_z)
        overload = obs["fz"] > 145.0 or abs(obs["tx"]) + abs(obs["ty"]) > 58.0
        if overload:
            self.failed = True
        if self.progress >= 1.0 and not self.failed:
            self.success = True
        return obs


def run_strategy(strategy: str, use_topology: bool, max_steps: int = 64) -> List[Dict[str, float]]:
    env = Virtual6AxisAssemblyEnv()
    topo = DualTopologyState3D()
    rows: List[Dict[str, float]] = []
    for step in range(max_steps):
        probe = env.observe(1.0)
        hint = topo.update(
            probe["err_x"],
            probe["err_y"],
            probe["fx"],
            probe["fy"],
            probe["fz"],
            probe["tx"],
            probe["ty"],
            probe["tz"],
            env.progress,
        )
        cmd_z = 1.0
        cmd_x = 0.0
        cmd_y = 0.0
        if use_topology:
            cmd_z = hint["push_scale"]
            dist_to_center = math.sqrt(env.pos_x**2 + env.pos_y**2)
            in_contact_zone = 0.37 < env.progress < 0.68 and dist_to_center > 0.035
            if hint["contact_lock"] > 0.32 and in_contact_zone:
                cmd_x = hint["release_bias_x"] * 0.55
                cmd_y = hint["release_bias_y"] * 0.55
            if dist_to_center < 0.08:
                cmd_x *= 0.4
                cmd_y *= 0.4
                cmd_z = max(cmd_z, 0.75)
        obs = env.step(cmd_z, cmd_x, cmd_y)
        rows.append(
            {
                "strategy": strategy,
                "step": step,
                "progress": env.progress,
                "pos_x": env.pos_x,
                "pos_y": env.pos_y,
                "fx": obs["fx"],
                "fy": obs["fy"],
                "fz": obs["fz"],
                "tx": obs["tx"],
                "ty": obs["ty"],
                "tz": obs["tz"],
                "contact_lock": hint["contact_lock"],
                "turns_x": hint["turns_x"],
                "turns_y": hint["turns_y"],
                "cmd_z": cmd_z,
                "cmd_x": cmd_x,
                "cmd_y": cmd_y,
                "failed": float(env.failed),
                "success": float(env.success),
            }
        )
        if env.failed or env.success:
            break
    return rows


def write_csv(rows: List[Dict[str, float]]) -> None:
    RESULTS.mkdir(exist_ok=True)
    fields = list(rows[0].keys())
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, float]], name: str) -> None:
    last = rows[-1]
    print(
        f"{name:28s} steps={len(rows):02d} success={bool(last['success'])} failed={bool(last['failed'])} "
        f"max_fz={max(r['fz'] for r in rows):.1f}N max_torque={max(abs(r['tx']) + abs(r['ty']) for r in rows):.1f} final_progress={last['progress']:.2f}"
    )


def main() -> None:
    baseline = run_strategy("reactive", False)
    topology = run_strategy("six_axis_topology", True)
    rows = baseline + topology
    write_csv(rows)
    summarize(baseline, "reactive hard-push")
    summarize(topology, "6-axis topology recovery")
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
