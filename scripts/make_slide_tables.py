#!/usr/bin/env python3
"""Quick slide-ready PNG tables (architecture spec + old-vs-new comparison)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/Users/estebanfelixmoran/Desktop/Purdue/Medical_Imaging/3D-Medical-Segment-Anything/outputs/two_query_experiment_figures/slide_tables")
OUT.mkdir(parents=True, exist_ok=True)

HEADER_BG = "#1f2937"
HEADER_FG = "white"
ROW_ALT = "#f0f3f6"
ROW_MAIN = "white"
BORDER = "#d7dde2"


def make_table(rows, col_labels, out_name, col_widths=None, title=None, figsize=(10, None)):
    n = len(rows)
    h = 0.55 + 0.42 * n + (0.5 if title else 0)
    fig, ax = plt.subplots(figsize=(figsize[0], h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", pad=14, loc="left", family="DejaVu Sans")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="left",
                    colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if r == 0:
            cell.set_facecolor(HEADER_BG); cell.set_text_props(color=HEADER_FG, fontweight="bold")
        else:
            cell.set_facecolor(ROW_ALT if r % 2 == 0 else ROW_MAIN)
        cell.PAD = 0.03
    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / out_name)


# ---------- Table 1: Architecture spec ----------
make_table(
    rows=[
        ["Source resolution", "Varies (native scanner res.)"],
        ["Resampled spacing", "0.167 × 0.200 × 0.160 mm"],
        ["Model input grid", "192 × 128 × 160 voxels, 1 channel"],
        ["Encoder", "Frozen Swin backbone, 10,690,434 params"],
        ["Encoder feature levels", "48 / 48 / 96 / 192 / 384 channels (L0→L4)"],
        ["Decoder canvas widths", "48 / 48 / 96 / 192 / 384 (unprojected, no bottleneck)"],
        ["Upsampling mechanism", "Trilinear interp. + 1×1×1 conv + concat + residual block"],
        ["Query embedding dim", "32, ×2 independent queries (brain, lesion)"],
        ["Decoder params (trainable)", "4,377,393"],
        ["Total params", "15,067,827  (29.1% trainable)"],
    ],
    col_labels=["Property", "Value"],
    out_name="architecture_spec_table.png",
    col_widths=[0.34, 0.66],
    title="Dual-Query Decoder — Architecture Specification",
)

# ---------- Table 2: Old vs new ----------
make_table(
    rows=[
        ["Decoder parameters", "182,081", "4,377,393  (×24)"],
        ["Canvas width, level 0", "16 (narrowed)", "48 (unreduced)"],
        ["Canvas widths, levels 1–4", "32 (compressed, shared)", "48 / 96 / 192 / 384 (native)"],
        ["Upsampling mechanism", "Single fusion pass", "Real skip-connected chain"],
        ["Queries", "1 (brain only)", "2 (brain + lesion, shared trunk)"],
        ["CAMRI test Dice", "0.9888", "0.9912  (+0.0024)"],
        ["Mouse test Dice", "0.9795", "0.9830  (+0.0035)"],
        ["Lesion segmentation", "Not supported", "0.8362 Dice  (new)"],
    ],
    col_labels=["Property", "Two weeks ago", "Current"],
    out_name="old_vs_new_table.png",
    col_widths=[0.32, 0.32, 0.36],
    title="Two Weeks Ago → Now",
)
