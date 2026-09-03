#!/usr/bin/env python3
"""Collect all 4 points of the embedding_dim/level0_width sweep into one
table + trend figure. Run only after all 4 train+evaluate pairs finish.
No invented numbers -- everything here is read back from files those
scripts wrote.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "outputs/level0_width_sweep"
WIDTHS = (32, 48, 64, 96)


def rows(p):
    return list(csv.DictReader(open(p)))


def main():
    records = []
    for emb in WIDTHS:
        d = SWEEP_ROOT / f"emb{emb}"
        summary = json.loads((d / "training/summary.json").read_text())
        native = rows(d / "native_metrics.csv")
        row = {"embedding_dim": emb, "level0_width": summary["level0_width"],
               "decoder_parameters": summary["decoder_parameters"],
               "selected_epoch": summary["selected_epoch"], "epochs_run": summary["epochs_run"],
               "best_balanced_validation_dice": summary["best_balanced_validation_dice"]}
        for r in native:
            if r["condition"] == "sweep_filtered":
                row[f"{r['domain'].lower()}_dice"] = float(r["dice"])
                row[f"{r['domain'].lower()}_hd95_mm"] = float(r["hd95_mm"])
                row[f"{r['domain'].lower()}_assd_mm"] = float(r["assd_mm"])
            if r["condition"] == "baseline_filtered":
                row[f"{r['domain'].lower()}_baseline_dice"] = float(r["dice"])
        records.append(row)

    with (SWEEP_ROOT / "sweep_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    embs = [r["embedding_dim"] for r in records]

    ax[0].plot(embs, [r["camri_dice"] for r in records], "o-", label="CAMRI")
    ax[0].plot(embs, [r["mouse_dice"] for r in records], "o-", label="Mouse")
    ax[0].axhline(records[0]["camri_baseline_dice"], color="C0", ls="--", lw=1, label="current best (CAMRI)")
    ax[0].axhline(records[0]["mouse_baseline_dice"], color="C1", ls="--", lw=1, label="current best (Mouse)")
    ax[0].set(xlabel="embedding_dim (= level0_width)", ylabel="Native test Dice", title="Dice vs. decoder width")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].plot(embs, [r["camri_hd95_mm"] for r in records], "o-", label="CAMRI")
    ax[1].plot(embs, [r["mouse_hd95_mm"] for r in records], "o-", label="Mouse")
    ax[1].set(xlabel="embedding_dim (= level0_width)", ylabel="HD95 (mm)", title="HD95 vs. decoder width")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    ax[2].plot(embs, [r["decoder_parameters"] for r in records], "o-", color="green")
    ax[2].set(xlabel="embedding_dim (= level0_width)", ylabel="Decoder parameters", title="Model size vs. width")
    ax[2].grid(alpha=.3)

    fig.suptitle("embedding_dim / level0_width sweep (unreduced final stage, matching the paper's convention)")
    fig.savefig(SWEEP_ROOT / "sweep_trend.png", dpi=180)
    plt.close(fig)

    print(json.dumps(records, indent=2))
    print("wrote", SWEEP_ROOT / "sweep_summary.csv", "and", SWEEP_ROOT / "sweep_trend.png")


if __name__ == "__main__":
    main()
