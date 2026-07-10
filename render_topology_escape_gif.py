"""Optional GIF renderer for the virtual 6-axis contact demo.

This script directly executes the current Python 6-axis virtual contact
simulation in memory and renders results/topology_escape.gif. It does not read a
pre-generated CSV file.

Optional dependencies:
    pip install matplotlib pillow
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This optional GIF renderer requires matplotlib and pillow.\n"
        "Install them with: py -m pip install matplotlib pillow"
    ) from exc

from animate_virtual_6axis_demo import run_strategy


ROOT = Path(__file__).resolve().parent
GIF_PATH = ROOT / "results" / "topology_escape.gif"


def split_traces() -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    baseline = run_strategy("reactive", False)
    topology = run_strategy("six_axis_topology", True)
    return baseline, topology


def row_at(series: List[Dict[str, float]], idx: int) -> Dict[str, float]:
    return series[min(idx, len(series) - 1)]


def main() -> None:
    baseline, topology = split_traces()
    rows = baseline + topology
    frames = max(len(baseline), len(topology))
    max_fz = max(float(row["fz"]) for row in rows) * 1.18

    fig, (ax_space, ax_force) = plt.subplots(1, 2, figsize=(11, 5), dpi=110)
    fig.suptitle("TopoEmbodiedLoop virtual 6-axis contact recovery", fontsize=13, fontweight="bold")

    ax_space.set_xlim(-0.28, 0.28)
    ax_space.set_ylim(-0.28, 0.28)
    ax_space.set_aspect("equal", adjustable="box")
    ax_space.set_title("Top-view contact trajectory")
    ax_space.set_xlabel("x lateral offset")
    ax_space.set_ylabel("y lateral offset")
    ax_space.grid(True, linestyle="--", alpha=0.35)
    hole = plt.Circle((0.0, 0.0), 0.05, color="#2ecc71", fill=False, linewidth=2, label="target tolerance")
    jam = plt.Circle((0.0, 0.0), 0.20, color="#e74c3c", fill=False, linestyle=":", linewidth=1.8, label="virtual jam zone")
    ax_space.add_patch(hole)
    ax_space.add_patch(jam)
    baseline_line, = ax_space.plot([], [], color="#e74c3c", linewidth=2.5, label="hard-push")
    topo_line, = ax_space.plot([], [], color="#2ecc71", linewidth=2.5, label="topology recovery")
    baseline_dot, = ax_space.plot([], [], "o", color="#e74c3c", markersize=7)
    topo_dot, = ax_space.plot([], [], "o", color="#2ecc71", markersize=7)
    status_text = ax_space.text(0.0, -0.255, "", ha="center", fontsize=9)
    ax_space.legend(loc="upper right", fontsize=8)

    ax_force.set_xlim(0, frames)
    ax_force.set_ylim(0, max_fz)
    ax_force.set_title("Synthetic Fz and contact-lock traces")
    ax_force.set_xlabel("simulation step")
    ax_force.set_ylabel("Fz / N")
    ax_force.grid(True, linestyle="--", alpha=0.35)
    ax_lock = ax_force.twinx()
    ax_lock.set_ylim(0, 1.0)
    ax_lock.set_ylabel("contact_lock")
    baseline_fz, = ax_force.plot([], [], color="#e74c3c", linewidth=2, label="hard-push Fz")
    topo_fz, = ax_force.plot([], [], color="#2ecc71", linewidth=2, label="topology Fz")
    topo_lock, = ax_lock.plot([], [], color="#f39c12", linewidth=2, linestyle="--", label="topology lock")
    ax_force.axhline(145.0, color="#c0392b", linestyle=":", linewidth=1.5)
    ax_force.text(1, 147.0, "overload threshold", color="#c0392b", fontsize=8)
    lines = [baseline_fz, topo_fz, topo_lock]
    ax_force.legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=8)

    def update(frame: int):
        b = baseline[: min(frame + 1, len(baseline))]
        t = topology[: min(frame + 1, len(topology))]
        bx = [float(row["pos_x"]) for row in b]
        by = [float(row["pos_y"]) for row in b]
        tx = [float(row["pos_x"]) for row in t]
        ty = [float(row["pos_y"]) for row in t]
        baseline_line.set_data(bx, by)
        topo_line.set_data(tx, ty)
        baseline_dot.set_data([bx[-1]], [by[-1]])
        topo_dot.set_data([tx[-1]], [ty[-1]])

        x_b = [float(row["step"]) for row in b]
        x_t = [float(row["step"]) for row in t]
        baseline_fz.set_data(x_b, [float(row["fz"]) for row in b])
        topo_fz.set_data(x_t, [float(row["fz"]) for row in t])
        topo_lock.set_data(x_t, [float(row["contact_lock"]) for row in t])

        current_b = row_at(baseline, frame)
        current_t = row_at(topology, frame)
        status_text.set_text(
            f"hard-push: progress {float(current_b['progress']):.2f}, failed {bool(current_b['failed'])} | "
            f"topology: progress {float(current_t['progress']):.2f}, success {bool(current_t['success'])}"
        )
        return baseline_line, topo_line, baseline_dot, topo_dot, baseline_fz, topo_fz, topo_lock, status_text

    ani = FuncAnimation(fig, update, frames=frames, interval=140, blit=False)
    GIF_PATH.parent.mkdir(exist_ok=True)
    ani.save(str(GIF_PATH), writer=PillowWriter(fps=7))
    plt.close(fig)
    print(f"Wrote {GIF_PATH}")


if __name__ == "__main__":
    main()
