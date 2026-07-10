"""Render a presentation-friendly GIF from the virtual 6-axis trace.

This renderer directly executes the current Python 6-axis virtual contact
simulation in memory, then creates a side-by-side story GIF: hard-push failure
vs topology-aware recovery. It does not read a pre-generated CSV file.

Optional dependencies:
    pip install matplotlib pillow
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This optional story GIF renderer requires matplotlib and pillow.\n"
        "Install them with: py -m pip install matplotlib pillow"
    ) from exc

from animate_virtual_6axis_demo import run_strategy


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parent
GIF_PATH = ROOT / "results" / "topology_escape_story.gif"


def build_traces() -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    return run_strategy("reactive", False), run_strategy("six_axis_topology", True)


def row_at(rows: List[Dict[str, float]], idx: int) -> Dict[str, float]:
    return rows[min(idx, len(rows) - 1)]


def draw_bar(ax, x: float, y: float, value: float, max_value: float, color: str, label: str) -> None:
    label_width = 0.15
    bar_width = 0.46
    height = 0.038
    bar_x = x + label_width
    ax.text(x, y + 0.006, label, fontsize=8, color="#333", ha="left", va="bottom")
    ax.add_patch(Rectangle((bar_x, y), bar_width, height, fill=False, edgecolor="#666", linewidth=0.9))
    filled = max(0.0, min(1.0, value / max_value)) * bar_width
    ax.add_patch(Rectangle((bar_x, y), filled, height, color=color, alpha=0.85))
    ax.text(bar_x + bar_width + 0.018, y + 0.006, f"{value:.1f}", fontsize=8, color="#333", ha="left", va="bottom")


def draw_panel(ax, rows: List[Dict[str, float]], frame: int, title: str, color: str, show_release_arrow: bool) -> None:
    row = row_at(rows, frame)
    past = rows[: min(frame + 1, len(rows))]
    progress = float(row["progress"])
    pos_x = float(row["pos_x"])
    pos_y = float(row["pos_y"])
    fz = float(row["fz"])
    torque = abs(float(row["tx"])) + abs(float(row["ty"]))
    lock = float(row["contact_lock"])
    failed = bool(row["failed"])
    success = bool(row["success"])

    ax.set_xlim(-0.33, 0.33)
    ax.set_ylim(-0.43, 0.40)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", color=color, pad=10)
    ax.add_patch(Circle((0, 0), 0.055, fill=False, color="#27ae60", linewidth=3))
    ax.add_patch(Circle((0, 0), 0.20, fill=False, color="#c0392b", linestyle=":", linewidth=2))
    ax.text(0.0, 0.066, "目标孔位", ha="center", fontsize=8, color="#27ae60", bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5})
    ax.text(0.0, 0.217, "接触风险区", ha="center", fontsize=8, color="#c0392b", bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5})

    xs = [float(r["pos_x"]) for r in past]
    ys = [float(r["pos_y"]) for r in past]
    ax.plot(xs, ys, color=color, linewidth=3, alpha=0.9)
    ax.scatter([pos_x], [pos_y], s=130, color=color, edgecolor="#1f2d3d", zorder=5)
    ax.plot([pos_x, pos_x], [pos_y, pos_y + 0.09], color="#1f2d3d", linewidth=4, solid_capstyle="round")

    if show_release_arrow and lock > 0.30 and not success:
        cmd_x = float(row["cmd_x"])
        cmd_y = float(row["cmd_y"])
        if abs(cmd_x) + abs(cmd_y) > 0.05:
            arrow = FancyArrowPatch(
                (pos_x, pos_y),
                (pos_x + cmd_x * 0.11, pos_y + cmd_y * 0.11),
                arrowstyle="->",
                mutation_scale=20,
                linewidth=3,
                color="#16a085",
            )
            ax.add_patch(arrow)
            ax.text(pos_x + cmd_x * 0.15, pos_y + cmd_y * 0.15, "降推+侧向", fontsize=9, color="#16a085", fontweight="bold", bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none", "pad": 1.5})

    draw_bar(ax, -0.31, -0.310, fz, 150.0, "#e67e22", "Fz")
    draw_bar(ax, -0.31, -0.365, torque, 65.0, "#9b59b6", "Torque")
    draw_bar(ax, -0.31, -0.420, lock, 1.0, "#f1c40f", "Lock")
    ax.text(-0.30, 0.345, f"插入进度 {progress:.2f}", fontsize=10, color="#333", bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5})

    if failed:
        ax.text(0.0, -0.245, "卡死：力矩过载", ha="center", fontsize=16, color="#c0392b", fontweight="bold", bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2.0})
    elif success:
        ax.text(0.0, -0.245, "成功插入", ha="center", fontsize=16, color="#27ae60", fontweight="bold", bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2.0})
    elif lock > 0.32 and show_release_arrow and abs(float(row["cmd_x"])) + abs(float(row["cmd_y"])) > 0.05:
        ax.text(0.0, -0.245, "降推 -> 侧向释放", ha="center", fontsize=11, color="#16a085", fontweight="bold", bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5})
    elif lock > 0.32:
        ax.text(0.0, -0.245, "卡滞风险上升", ha="center", fontsize=11, color="#c0392b", fontweight="bold", bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5})
    else:
        ax.text(0.0, -0.245, "接近目标", ha="center", fontsize=11, color="#555", bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5})


def main() -> None:
    baseline, topology = build_traces()
    frames = max(len(baseline), len(topology))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=120)
    fig.patch.set_facecolor("white")
    fig.suptitle("硬推卡死，拓扑释放成功", fontsize=17, fontweight="bold", y=0.975)
    fig.text(0.5, 0.905, "由当前 Python 六轴虚拟接触仿真直接运行生成；非真机 benchmark", ha="center", fontsize=10, color="#555")
    fig.subplots_adjust(top=0.80, bottom=0.06, left=0.04, right=0.98, wspace=0.10)

    def update(frame: int):
        for ax in axes:
            ax.clear()
        draw_panel(axes[0], baseline, frame, "传统硬推", "#e74c3c", False)
        draw_panel(axes[1], topology, frame, "拓扑恢复", "#2ecc71", True)
        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=145, blit=False)
    GIF_PATH.parent.mkdir(exist_ok=True)
    ani.save(str(GIF_PATH), writer=PillowWriter(fps=7))
    plt.close(fig)
    print(f"Wrote {GIF_PATH}")


if __name__ == "__main__":
    main()
