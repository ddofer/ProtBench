"""Generate deterministic SVG figures for the ProtBench paper draft.

The figures are intentionally plain vector graphics built from the live task
registry. They are meant as manuscript schematics, not decorative artwork.
"""

from __future__ import annotations

import argparse
import html
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_tasks import TASKS  # noqa: E402


PALETTE = {
    "ink": "#1f2933",
    "muted": "#5f6b7a",
    "line": "#c9d2dc",
    "paper": "#fbfcfd",
    "blue": "#2f6f9f",
    "green": "#2f8f6f",
    "gold": "#b8841a",
    "red": "#b54b4b",
    "purple": "#6f5aa8",
    "gray": "#e8edf2",
}

TYPE_LABELS = {
    "binary": "Binary",
    "multiclass": "Multiclass",
    "multilabel": "Multilabel",
    "regression": "Regression",
    "retrieval": "Retrieval",
    "token_classification": "Residue",
}

VERIFIED_DATASETS = {
    "agemagician/NetSurfP-SS3",
    "SaProtHub/Dataset-Signal-Peptides",
    "data/disprot",
}

MAPPED_DATASETS = {
    "OATML-Markslab/ProteinGym_v1",
    "GrimSqueaker/cath43-eat",
    "SaProtHub/Dataset-AAV-FLIP",
    "SaProtHub/Dataset-Beta_Lactamase-PEER",
    "SaProtHub/Dataset-Thermostability-FLIP",
    "Synthyra/bernett_gold_ppi",
    "cradle-bio/tape-fluorescence",
    "hazemessam/meltome",
    "mila-intel/ProtST-BinaryLocalization",
    "proteinea/deeploc",
    "tattabio/scope40_test",
    "GrimSqueaker/ProFET_NP_SP_Cleaved",
    "GrimSqueaker/SignalP_Binary",
    "andrewdalpino/CAFA5",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


class Svg:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            "<defs>",
            "<marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"8\" refX=\"9\" refY=\"4\" orient=\"auto\" markerUnits=\"strokeWidth\"><path d=\"M0,0 L10,4 L0,8 Z\" fill=\"#6b7684\"/></marker>",
            "</defs>",
            f'<rect width="{width}" height="{height}" fill="{PALETTE["paper"]}"/>',
        ]

    def text(self, x: int, y: int, text: str, size: int = 18, weight: int = 400, fill: str | None = None, anchor: str = "start") -> None:
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill or PALETTE["ink"]}" text-anchor="{anchor}">{esc(text)}</text>'
        )

    def multiline(self, x: int, y: int, lines: list[str], size: int = 15, fill: str | None = None, line_height: int = 20, anchor: str = "start") -> None:
        for offset, line in enumerate(lines):
            self.text(x, y + offset * line_height, line, size=size, fill=fill, anchor=anchor)

    def rect(self, x: int, y: int, w: int, h: int, fill: str, stroke: str | None = None, rx: int = 8, width: int = 1) -> None:
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke or PALETTE["line"]}" stroke-width="{width}"/>'
        )

    def line(self, x1: int, y1: int, x2: int, y2: int, stroke: str | None = None, width: int = 2, arrow: bool = False) -> None:
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke or PALETTE["muted"]}" stroke-width="{width}" stroke-linecap="round"{marker}/>'
        )

    def circle(self, cx: int, cy: int, r: int, fill: str, stroke: str | None = None) -> None:
        self.parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke or PALETTE["line"]}" stroke-width="1"/>'
        )

    def save(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts) + "\n")


def provenance_tier(dataset: str) -> str:
    if dataset.startswith("data/"):
        return "Local build"
    if dataset in VERIFIED_DATASETS:
        return "Verified"
    if dataset in MAPPED_DATASETS:
        return "Mapped"
    return "Best-effort"


def draw_graphical_abstract(out_dir: Path) -> Path:
    svg = Svg(1280, 720)
    svg.text(60, 70, "ProtBench evaluates protein representations under one contract", 30, 700)
    svg.text(60, 105, "Same tasks, splits, probes, metrics, and result rows across model families", 18, fill=PALETTE["muted"])

    columns = [
        (70, 170, 250, 280, "Inputs", PALETTE["blue"], ["Protein sequences", "Pretrained PLMs", "k-mer / alignment baselines"]),
        (395, 150, 310, 320, "Evaluation contract", PALETTE["green"], ["Task registry", "Declared splits", "Primary metrics", "Seeded probes", "Provenance notes"]),
        (780, 170, 250, 280, "Evaluation modes", PALETTE["gold"], ["Frozen linear probe", "k-NN / HistGB", "LoRA / full fine-tune", "ProteinGym zero-shot"]),
        (1090, 190, 140, 240, "Outputs", PALETTE["purple"], ["CSV rows", "Long table", "Figures", "Caveats"]),
    ]
    for x, y, w, h, title, color, lines in columns:
        svg.rect(x, y, w, h, "#ffffff", stroke=color, rx=10, width=2)
        svg.text(x + 20, y + 42, title, 20, 700, fill=color)
        for i, line in enumerate(lines):
            svg.circle(x + 26, y + 82 + i * 38, 5, color)
            svg.text(x + 42, y + 88 + i * 38, line, 16)

    svg.line(320, 310, 395, 310, arrow=True)
    svg.line(705, 310, 780, 310, arrow=True)
    svg.line(1030, 310, 1090, 310, arrow=True)

    svg.rect(165, 535, 950, 95, "#f2f6f9", stroke=PALETTE["line"], rx=12)
    svg.text(195, 572, "Reader takeaway", 18, 700, fill=PALETTE["red"])
    svg.text(195, 602, "A model comparison should identify the biological task, the evaluation question, and the baseline it must clear.", 18)
    svg.save(out_dir / "protbench_graphical_abstract.svg")
    return out_dir / "protbench_graphical_abstract.svg"


def draw_task_landscape(out_dir: Path) -> Path:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for cfg in TASKS.values():
        counts[cfg.problem_type][provenance_tier(cfg.dataset)] += 1

    tiers = ["Verified", "Mapped", "Local build", "Best-effort"]
    tier_colors = {
        "Verified": PALETTE["green"],
        "Mapped": PALETTE["blue"],
        "Local build": PALETTE["gold"],
        "Best-effort": PALETTE["red"],
    }
    rows = sorted(TYPE_LABELS, key=lambda key: TYPE_LABELS[key])

    svg = Svg(1280, 720)
    svg.text(60, 70, "Task coverage and provenance tiers", 30, 700)
    svg.text(60, 105, "Counts are generated from benchmark_tasks.py; provenance tiers summarize docs/DATASETS.md status.", 18, fill=PALETTE["muted"])

    x0, y0 = 290, 165
    cell_w, cell_h = 190, 64
    svg.text(60, 145, "Problem type", 16, 700, fill=PALETTE["muted"])
    for col, tier in enumerate(tiers):
        svg.text(x0 + col * cell_w + cell_w // 2, 145, tier, 16, 700, fill=tier_colors[tier], anchor="middle")

    for row_idx, problem_type in enumerate(rows):
        y = y0 + row_idx * cell_h
        total = sum(counts[problem_type].values())
        svg.text(60, y + 38, f"{TYPE_LABELS[problem_type]} ({total})", 17, 700)
        for col, tier in enumerate(tiers):
            x = x0 + col * cell_w
            value = counts[problem_type][tier]
            fill = "#ffffff" if value == 0 else tier_colors[tier]
            svg.rect(x, y, cell_w - 18, cell_h - 14, fill, stroke=PALETTE["line"], rx=8, width=1)
            svg.text(x + (cell_w - 18) // 2, y + 34, str(value), 22, 700, fill="#ffffff" if value else PALETTE["muted"], anchor="middle")

    legend_y = 590
    svg.rect(70, legend_y - 32, 1140, 82, "#ffffff", stroke=PALETTE["line"], rx=10)
    legend_items = [
        ("Verified", "Checked against upstream release or local build details"),
        ("Mapped", "Linked to an original benchmark paper but not fully reverified"),
        ("Best-effort", "Third-party rehost or source still needs manual confirmation"),
    ]
    x = 100
    for tier, desc in legend_items:
        svg.circle(x, legend_y, 7, tier_colors[tier])
        svg.text(x + 18, legend_y + 5, f"{tier}: {desc}", 14)
        x += 360

    svg.save(out_dir / "task_provenance_landscape.svg")
    return out_dir / "task_provenance_landscape.svg"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="paper/figures", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in (draw_graphical_abstract(out_dir), draw_task_landscape(out_dir)):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())