"""Generate simple SVG figures for the Word document."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CSV_PATH = RESULTS / "summary.csv"
SVG_PATH = RESULTS / "topo_embodied_loop_comparison.svg"


def read_rows():
    with CSV_PATH.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bar_chart_group(rows):
    metrics = [
        ("success_rate", "Success rate (%)", 100.0),
        ("avg_max_force", "Peak force", 1.0),
        ("avg_impulse", "Impulse", 1.0),
        ("avg_jam_steps", "Jam steps", 1.0),
    ]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    width = 980
    height = 700
    margin_left = 70
    margin_top = 70
    panel_w = 420
    panel_h = 240
    gap_x = 60
    gap_y = 70

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="490" y="34" text-anchor="middle" font-size="22" font-family="Arial" font-weight="bold">TopoEmbodiedLoop closed-loop comparison</text>',
        '<text x="490" y="58" text-anchor="middle" font-size="13" font-family="Arial" fill="#555">reactive vs reasoner_guard vs full_loop</text>',
    ]

    for mi, (key, title, scale) in enumerate(metrics):
        row = mi // 2
        col = mi % 2
        x0 = margin_left + col * (panel_w + gap_x)
        y0 = margin_top + row * (panel_h + gap_y)
        values = [float(r[key]) * scale for r in rows]
        ymax = max(values) * 1.18 if max(values) > 0 else 1.0
        if key == "success_rate":
            ymax = 100.0
        parts.append(f'<text x="{x0 + panel_w / 2}" y="{y0 - 14}" text-anchor="middle" font-size="16" font-family="Arial" font-weight="bold">{title}</text>')
        parts.append(f'<line x1="{x0}" y1="{y0 + panel_h}" x2="{x0 + panel_w}" y2="{y0 + panel_h}" stroke="#333"/>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + panel_h}" stroke="#333"/>')
        for tick in range(5):
            tv = ymax * tick / 4
            ty = y0 + panel_h - panel_h * tick / 4
            parts.append(f'<line x1="{x0 - 4}" y1="{ty}" x2="{x0 + panel_w}" y2="{ty}" stroke="#eee"/>')
            parts.append(f'<text x="{x0 - 8}" y="{ty + 4}" text-anchor="end" font-size="11" font-family="Arial" fill="#555">{tv:.0f}</text>')
        bar_w = 74
        spacing = 55
        start_x = x0 + 65
        for i, (r, v) in enumerate(zip(rows, values)):
            bh = 0 if ymax <= 0 else panel_h * v / ymax
            bx = start_x + i * (bar_w + spacing)
            by = y0 + panel_h - bh
            parts.append(f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" fill="{colors[i]}" rx="3"/>')
            label = f'{v:.1f}' if key != "success_rate" else f'{v:.1f}%'
            parts.append(f'<text x="{bx + bar_w / 2}" y="{by - 6}" text-anchor="middle" font-size="12" font-family="Arial">{label}</text>')
            name = r["strategy"].replace("_", "_")
            parts.append(f'<text x="{bx + bar_w / 2}" y="{y0 + panel_h + 20}" text-anchor="middle" font-size="11" font-family="Arial" transform="rotate(-18 {bx + bar_w / 2},{y0 + panel_h + 20})">{name}</text>')
    parts.append('</svg>')
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main():
    rows = read_rows()
    bar_chart_group(rows)
    print(f"Wrote {SVG_PATH}")


if __name__ == "__main__":
    main()
