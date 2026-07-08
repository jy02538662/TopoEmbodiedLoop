"""Small contact-rich insertion simulator for TopoEmbodiedLoop."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

from modules.common import STATE_ID, clip


@dataclass
class Scene:
    scene_id: int
    preferred_release: int
    difficulty: float


@dataclass
class RobotState:
    contact_state: str
    progress: float
    lateral_error: float
    force: float
    tightness: float
    released_once: bool = False
    recaptured_once: bool = False
    last_release_dir: int = 0
    jam_streak: int = 0


def make_scene(rng: random.Random, n_families: int = 8) -> Scene:
    scene_id = rng.randrange(n_families)
    return Scene(
        scene_id=scene_id,
        preferred_release=1 if scene_id % 2 == 0 else -1,
        difficulty=0.42 + 0.48 * rng.random(),
    )


def initial_state(scene: Scene, rng: random.Random) -> RobotState:
    return RobotState(
        contact_state="contact",
        progress=0.08 + 0.10 * rng.random(),
        lateral_error=0.42 + 0.20 * scene.difficulty,
        force=12.0 + 8.0 * scene.difficulty,
        tightness=0.20 + 0.22 * scene.difficulty,
    )


def sensor_features(state: RobotState, scene: Scene, rng: random.Random) -> Dict[str, float]:
    vibration = clip(0.18 + 0.65 * state.tightness + rng.uniform(-0.08, 0.08), 0.0, 1.0)
    optical_lock = clip(1.0 - state.lateral_error + rng.uniform(-0.08, 0.08), 0.0, 1.0)
    audio_quality = clip(1.0 - 0.35 * vibration + rng.uniform(-0.05, 0.05), 0.0, 1.0)
    optical_quality = clip(optical_lock - 0.15 * scene.difficulty + rng.uniform(-0.05, 0.05), 0.0, 1.0)
    return {
        "force": state.force,
        "vibration": vibration,
        "optical_lock": optical_lock,
        "audio_quality": audio_quality,
        "optical_quality": optical_quality,
    }


def apply_action(state: RobotState, scene: Scene, action: str, release_dir: int, scale: float, rng: random.Random) -> None:
    previous = state.contact_state
    if action == "insert":
        state.progress += scale * (0.045 * (1.0 - 0.55 * state.tightness) - 0.004 * scene.difficulty)
        state.lateral_error -= scale * 0.031 * (1.0 - 0.45 * state.tightness)
        state.force += scale * (4.2 + 6.2 * scene.difficulty + 8.2 * state.tightness)
        state.tightness += 0.030 + 0.035 * scene.difficulty
        if state.force > 66.0 or state.tightness > 0.68:
            state.contact_state = "slip"
        if state.force > 88.0 or state.tightness > 0.84:
            state.contact_state = "jam"
    elif action == "release":
        state.released_once = True
        state.last_release_dir = release_dir
        match = 1.0 if release_dir == scene.preferred_release else -0.45
        state.force += -18.0 * max(match, 0.0) + 10.0 * max(-match, 0.0)
        state.tightness += -0.22 * max(match, 0.0) + 0.11 * max(-match, 0.0)
        state.lateral_error += 0.030 * max(match, 0.0) + 0.02 * max(-match, 0.0)
        state.contact_state = "release" if match > 0 else "jam"
    elif action == "recapture":
        state.recaptured_once = True
        state.lateral_error -= 0.070 * (1.0 - 0.42 * state.tightness)
        state.progress += 0.032 * (1.0 - 0.45 * state.tightness)
        state.force += 1.4 * scene.difficulty - 5.0 * (1.0 - state.tightness)
        state.tightness -= 0.065
        state.contact_state = "recapture" if state.lateral_error > 0.12 else "contact"
    elif action == "retreat":
        state.force -= 18.0
        state.tightness -= 0.22
        state.lateral_error += 0.06
        state.progress = max(0.0, state.progress - 0.035)
        state.contact_state = "release"

    state.progress = clip(state.progress, 0.0, 1.08)
    state.lateral_error = clip(state.lateral_error, 0.02, 0.85)
    state.force = clip(state.force + rng.uniform(-2.0, 2.0), 0.0, 150.0)
    state.tightness = clip(state.tightness + rng.uniform(-0.015, 0.015), 0.0, 1.0)

    if state.contact_state == "jam":
        state.jam_streak += 1
    elif previous == "jam" and state.contact_state == "release":
        state.jam_streak = max(0, state.jam_streak - 2)
    else:
        state.jam_streak = max(0, state.jam_streak - 1)
