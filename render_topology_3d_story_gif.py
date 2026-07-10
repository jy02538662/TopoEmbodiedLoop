"""Render a 3D presentation GIF from the virtual 6-axis contact trace.

This renderer directly executes the current Python 6-axis virtual contact
simulation in memory, then visualizes the trace as a side-by-side 3D insertion
scene: hard-push failure vs topology-aware recovery. It does not read a
pre-generated CSV file.

Optional dependencies:
    pip install matplotlib pillow
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This optional 3D GIF renderer requires matplotlib and pillow.\n"
        "Install them with: py -m pip install matplotlib pillow"
    ) from exc

from animate_virtual_6axis_demo import run_strategy


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parent
GIF_PATH = ROOT / "results" / "topology_escape_3d_story.gif"


def build_traces() -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    return run_strategy("reactive", False), run_strategy("six_axis_topology", True)


def row_at(rows: List[Dict[str, float]], idx: int) -> Dict[str, float]:
    return rows[min(idx, len(rows) - 1)]


def draw_socket(ax) -> None:
    ax.plot([0, 0], [0, 0], [0, 1.05], color="#95a5a6", linestyle="--", linewidth=1.4)
    for z, radius, color, alpha in [(0.38, 0.20, "#c0392b", 0.22), (0.68, 0.10, "#f39c12", 0.20), (1.00, 0.055, "#27ae60", 0.35)]:
        xs = []
        ys = []
        zs = []
        for i in range(80):
            t = 2 * math.pi * i / 79
            xs.append(radius * math.cos(t))
            ys.append(radius * math.sin(t))
            zs.append(z)
        ax.plot(xs, ys, zs, color=color, linewidth=2.0, alpha=0.95)
        segments = [[(xs[i], ys[i], 0.0), (xs[i], ys[i], z)] for i in range(0, 80, 10)]
        ax.add_collection3d(Line3DCollection(segments, colors=color, linewidths=0.45, alpha=alpha))


def draw_force_text(ax, row: Dict[str, float], x: float, y: float) -> None:
    fz = float(row["fz"])
    torque = abs(float(row["tx"])) + abs(float(row["ty"]))
    lock = float(row["contact_lock"])
    ax.text2D(x, y, f"Fz {fz:5.1f} N", transform=ax.transAxes, fontsize=9, color="#e67e22", bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5})
    ax.text2D(x, y - 0.045, f"Torque {torque:4.1f}", transform=ax.transAxes, fontsize=9, color="#8e44ad", bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5})
    ax.text2D(x, y - 0.090, f"Lock {lock:.2f}", transform=ax.transAxes, fontsize=9, color="#d35400", bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5})


def draw_scene(ax, rows: List[Dict[str, float]], frame: int, title: str, color: str, show_release: bool) -> None:
    row = row_at(rows, frame)
    past = rows[: min(frame + 1, len(rows))]
    x = float(row["pos_x"])
    y = float(row["pos_y"])
    z = float(row["progress"])
    failed = bool(row["failed"])
    success = bool(row["success"])
    lock = float(row["contact_lock"])

    ax.clear()
    ax.set_title(title, fontsize=13, fontweight="bold", color=color, pad=12)
    ax.set_xlim(-0.28, 0.28)
    ax.set_ylim(-0.28, 0.28)
    ax.set_zlim(0.0, 1.05)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("")
    ax.view_init(elev=24, azim=-55)
    ax.grid(True, alpha=0.25)
    draw_socket(ax)

    xs = [float(r["pos_x"]) for r in past]
    ys = [float(r["pos_y"]) for r in past]
    zs = [float(r["progress"]) for r in past]
    ax.plot(xs, ys, zs, color=color, linewidth=3.0)
    ax.scatter([x], [y], [z], s=70, color=color, edgecolor="#1f2d3d", depthshade=True)
    ax.plot([x, x], [y, y], [max(0.0, z - 0.16), z + 0.05], color="#2c3e50", linewidth=5.0)

    if show_release and lock > 0.30 and not success:
        cmd_x = float(row["cmd_x"])
        cmd_y = float(row["cmd_y"])
        if abs(cmd_x) + abs(cmd_y) > 0.05:
            ax.quiver(x, y, z, cmd_x * 0.10, cmd_y * 0.10, 0.0, color="#16a085", linewidth=2.5, arrow_length_ratio=0.35)
            ax.text(x + cmd_x * 0.15, y + cmd_y * 0.15, z + 0.05, "降推+侧向", color="#16a085", fontsize=9, fontweight="bold", bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.5})

    draw_force_text(ax, row, 0.03, 0.78)
    if failed:
        ax.text2D(0.42, 0.78, "卡死：力矩过载", transform=ax.transAxes, fontsize=13, color="#c0392b", fontweight="bold", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2.0})
    elif success:
        ax.text2D(0.48, 0.78, "成功插入", transform=ax.transAxes, fontsize=13, color="#27ae60", fontweight="bold", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2.0})
    elif lock > 0.32 and show_release and abs(float(row["cmd_x"])) + abs(float(row["cmd_y"])) > 0.05:
        ax.text2D(0.38, 0.78, "降推 -> 侧向释放", transform=ax.transAxes, fontsize=10, color="#16a085", fontweight="bold", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5})
    elif lock > 0.32:
        ax.text2D(0.42, 0.78, "卡滞风险上升", transform=ax.transAxes, fontsize=10, color="#c0392b", fontweight="bold", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5})


def main() -> None:
    baseline, topology = build_traces()
    frames = max(len(baseline), len(topology))
    fig = plt.figure(figsize=(12, 6), dpi=115)
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")
    fig.patch.set_facecolor("white")
    fig.suptitle("硬推卡死，拓扑释放成功", fontsize=16, fontweight="bold", y=0.975)
    fig.text(0.5, 0.905, "由当前 Python 六轴虚拟接触仿真直接运行生成；非真机 benchmark", ha="center", fontsize=10, color="#555")
    fig.subplots_adjust(top=0.80, bottom=0.04, left=0.03, right=0.97, wspace=0.02)

    def update(frame: int):
        draw_scene(ax_left, baseline, frame, "传统硬推", "#e74c3c", False)
        draw_scene(ax_right, topology, frame, "拓扑恢复", "#2ecc71", True)
        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=145, blit=False)
    GIF_PATH.parent.mkdir(exist_ok=True)
    ani.save(str(GIF_PATH), writer=PillowWriter(fps=7))
    plt.close(fig)
    print(f"Wrote {GIF_PATH}")


if __name__ == "__main__":
    main()
