"""Shared state definitions for TopoEmbodiedLoop."""

from __future__ import annotations

STATE_NAMES = ["free", "contact", "slip", "jam", "release", "recapture"]
STATE_ID = {name: idx for idx, name in enumerate(STATE_NAMES)}
BAD_STATE_IDS = [STATE_ID["slip"], STATE_ID["jam"], STATE_ID["release"]]


def normalize(values):
    total = float(sum(values))
    if total <= 1e-12:
        return [1.0 / len(values)] * len(values)
    return [float(v) / total for v in values]


def argmax(values):
    return max(range(len(values)), key=lambda i: values[i])


def clip(x, lo, hi):
    return max(lo, min(hi, x))
