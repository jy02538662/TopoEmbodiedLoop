"""TopoEmbodiedLoop open-source demo.

A compact integration of five research prototypes:

- DualWave idea: reliability-gated multimodal perception.
- TopoWave idea: contact-state perception.
- TopoClosedLoop idea: temporal reasoner + emergency guard.
- TideMemory idea: episodic memory as policy prior.
- VTEC idea: low-force release and recapture control.

Run:
    python run_demo.py
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import mean
from typing import Dict, List

from modules.contact_controller import ContactController
from modules.contact_reasoner import ContactPerception, TopoReasoner
from modules.reliability_gate import ReliabilityGate
from modules.simulator import RobotState, apply_action, initial_state, make_scene, sensor_features
from modules.topo_guard import TopoGuard
from modules.topo_memory import TopoMemory
from modules.dual_topology import DualTopologyState


MAX_STEPS = 110
DT = 0.05
SUCCESS_PROGRESS = 0.64
SUCCESS_LATERAL = 0.18
HARD_FORCE = 132.0
IMPULSE_LIMIT = 74.0


def one_hot_noisy_state(state: RobotState) -> List[float]:
    from modules.common import STATE_ID, STATE_NAMES, normalize

    probs = [0.06] * len(STATE_NAMES)
    probs[STATE_ID[state.contact_state]] += 0.72
    if state.force > 70:
        probs[STATE_ID["jam"]] += 0.12
    if state.tightness > 0.65:
        probs[STATE_ID["slip"]] += 0.08
    return normalize(probs)


def run_episode(strategy: str, scene, seed: int, memory: TopoMemory | None) -> Dict[str, object]:
    rng = random.Random(seed)
    state = initial_state(scene, rng)
    gate = ReliabilityGate()
    perception = ContactPerception()
    reasoner = TopoReasoner()
    guard = TopoGuard(dt=DT, hard_force=HARD_FORCE, impulse_limit=IMPULSE_LIMIT)
    controller = ContactController()
    topology = DualTopologyState()

    impulse = 0.0
    max_force = state.force
    jam_steps = 0
    emergency_count = 0
    last_sig = None

    for step in range(MAX_STEPS):
        prev_force = state.force
        feat = sensor_features(state, scene, rng)
        reliability = gate.update(feat["audio_quality"], feat["optical_quality"])
        force_slope = (state.force - prev_force) / DT
        optical_pred = perception.predict(state.force, force_slope, feat["vibration"], feat["optical_lock"])
        audio_probs = one_hot_noisy_state(state)
        fused_probs = gate.fuse(audio_probs, optical_pred["state_probs"], reliability)

        topo_hint = topology.update(
            state.lateral_error,
            state.force,
            state.tightness,
            1.0,
            state.progress,
        )

        if strategy == "reactive":
            action, release_dir, action_scale = controller.reactive_action(fused_probs, rng)
            scale = action_scale
            memory_prior = {"risk_prior": 0.0, "release_bias": 0, "confidence": 0.0}
        else:
            belief_out = reasoner.update(fused_probs, reliability["combined"])
            last_sig = TopoMemory().signature(
                scene.scene_id,
                belief_out["belief"],
                state.force,
                state.tightness,
                state.progress,
                topo_hint,
            )
            if strategy == "full_loop" and memory is not None:
                memory_prior = memory.recall(last_sig)
            else:
                memory_prior = {"risk_prior": 0.0, "release_bias": 0, "confidence": 0.0}
            guard_out = guard.update(
                belief_out["belief"],
                state.force,
                max(state.tightness, topo_hint["contact_lock"]),
                belief_out["uncertainty"],
                reliability["combined"],
                memory_prior,
            )
            if guard_out["mode"] in {"retreat", "release", "slow_down", "reobserve"}:
                emergency_count += 1
            action, release_dir, action_scale = controller.select_action(
                guard_out["mode"], belief_out["belief"], memory_prior, topo_hint, rng
            )
            scale = min(guard_out["scale"], action_scale)

        apply_action(state, scene, action, release_dir, scale, rng)
        max_force = max(max_force, state.force)
        impulse += state.force * DT
        jam_steps += int(state.contact_state == "jam")

        success = state.progress >= SUCCESS_PROGRESS and state.lateral_error <= SUCCESS_LATERAL and max_force <= 108.0
        if success:
            if strategy == "full_loop" and memory is not None and last_sig is not None:
                memory.store(last_sig, True, state.last_release_dir)
            return {
                "strategy": strategy,
                "success": True,
                "failure": "success",
                "steps": step + 1,
                "max_force": max_force,
                "impulse": impulse,
                "jam_steps": jam_steps,
                "emergency_count": emergency_count,
            }

        if max_force > HARD_FORCE or state.jam_streak > 14 or impulse > IMPULSE_LIMIT:
            if strategy == "full_loop" and memory is not None and last_sig is not None:
                memory.store(last_sig, False, state.last_release_dir)
            failure = "force" if max_force > HARD_FORCE else "jam" if state.jam_streak > 14 else "impulse"
            return {
                "strategy": strategy,
                "success": False,
                "failure": failure,
                "steps": step + 1,
                "max_force": max_force,
                "impulse": impulse,
                "jam_steps": jam_steps,
                "emergency_count": emergency_count,
            }

    if strategy == "full_loop" and memory is not None and last_sig is not None:
        memory.store(last_sig, False, state.last_release_dir)
    return {
        "strategy": strategy,
        "success": False,
        "failure": "timeout",
        "steps": MAX_STEPS,
        "max_force": max_force,
        "impulse": impulse,
        "jam_steps": jam_steps,
        "emergency_count": emergency_count,
    }


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    failures = [row["failure"] for row in rows if not row["success"]]
    return {
        "strategy": rows[0]["strategy"],
        "episodes": len(rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
        "avg_steps": mean(float(row["steps"]) for row in rows),
        "avg_max_force": mean(float(row["max_force"]) for row in rows),
        "avg_impulse": mean(float(row["impulse"]) for row in rows),
        "avg_jam_steps": mean(float(row["jam_steps"]) for row in rows),
        "avg_emergency_count": mean(float(row["emergency_count"]) for row in rows),
        "failure_counts": "none" if not failures else ",".join(f"{f}:{failures.count(f)}" for f in sorted(set(failures))),
    }


def run_benchmark(episodes: int = 160, seed: int = 11) -> List[Dict[str, object]]:
    scene_rng = random.Random(seed)
    scenes = [make_scene(scene_rng) for _ in range(episodes)]
    strategies = ["reactive", "reasoner_guard", "full_loop"]
    memory = TopoMemory(k=7)
    summaries = []
    for strategy in strategies:
        rows = []
        for idx, scene in enumerate(scenes):
            rows.append(run_episode(strategy, scene, seed * 10000 + idx * 17, memory if strategy == "full_loop" else None))
        summaries.append(summarize(rows))
    return summaries


def write_results(summary: List[Dict[str, object]], root: Path) -> None:
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def print_summary(summary: List[Dict[str, object]]) -> None:
    print("\nTopoEmbodiedLoop open-source demo")
    print("=" * 102)
    print(f"{'strategy':<16} {'success':>8} {'force':>10} {'impulse':>10} {'jam_steps':>10} {'emergency':>10}  failures")
    print("-" * 102)
    for row in summary:
        print(
            f"{row['strategy']:<16} "
            f"{100 * float(row['success_rate']):7.1f}% "
            f"{float(row['avg_max_force']):10.2f} "
            f"{float(row['avg_impulse']):10.2f} "
            f"{float(row['avg_jam_steps']):10.2f} "
            f"{float(row['avg_emergency_count']):10.2f}  "
            f"{row['failure_counts']}"
        )
    base = summary[0]
    print("=" * 102)
    for row in summary[1:]:
        success_gain = 100 * (float(row["success_rate"]) - float(base["success_rate"]))
        force_drop = 100 * (float(base["avg_max_force"]) - float(row["avg_max_force"])) / float(base["avg_max_force"])
        jam_drop = 100 * (float(base["avg_jam_steps"]) - float(row["avg_jam_steps"])) / max(float(base["avg_jam_steps"]), 1e-9)
        print(f"{row['strategy']}: success +{success_gain:.1f} pts, peak force -{force_drop:.1f}%, jam steps -{jam_drop:.1f}%")


def main() -> None:
    root = Path(__file__).resolve().parent
    summary = run_benchmark()
    write_results(summary, root)
    print_summary(summary)


if __name__ == "__main__":
    main()
