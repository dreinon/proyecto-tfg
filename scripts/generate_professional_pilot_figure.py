"""Generate the external professional-pilot transfer figure from reviewed evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

CONDITIONS = (
    "x2-clean",
    "x2-moderate",
    "x2-strong",
    "x4-clean",
    "x4-moderate",
    "x4-strong",
)
CONDITION_LABELS = {
    "x2-clean": "x2 limpia",
    "x2-moderate": "x2 moderada",
    "x2-strong": "x2 fuerte",
    "x4-clean": "x4 limpia",
    "x4-moderate": "x4 moderada",
    "x4-strong": "x4 fuerte",
}
ACCEPTANCE_ORDER = ("acceptable", "acceptable-with-reservations", "rejected")
ACCEPTANCE_LABELS = {
    "acceptable": "Aceptable",
    "acceptable-with-reservations": "Con reservas",
    "rejected": "Rechazado",
}
ACCEPTANCE_COLORS = {
    "acceptable": "#2F6690",
    "acceptable-with-reservations": "#D09B2C",
    "rejected": "#9A4E3F",
}
ACCEPTANCE_HATCHES = {
    "acceptable": "",
    "acceptable-with-reservations": "//",
    "rejected": "xx",
}


def _comma(value: float, _position: float) -> str:
    return f"{value:g}".replace(".", ",")


def load_evidence(artifact_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = pd.read_csv(artifact_root / "paired-bootstrap.csv")
    paired = paired[
        paired["comparator_id"].eq("edsr-baseline-official-v1")
        & paired["method_id"].eq("edsr-smb-finetuned-v1")
    ].copy()
    expected = {(condition, metric) for condition in CONDITIONS for metric in ("psnr_y", "ssim_y")}
    observed = set(zip(paired["condition_id"], paired["metric"], strict=True))
    if len(paired) != 12 or observed != expected or not paired["sources"].eq(12).all():
        raise ValueError("paired pilot evidence does not contain the expected twelve comparisons")

    review = json.loads(
        (artifact_root / "professional-pilot-v1-qualitative-review.json").read_text(
            encoding="utf-8"
        )
    )
    assessments = pd.DataFrame(review["assessments"])
    if (
        review.get("case_count") != 12
        or len(assessments) != 12
        or set(assessments["acceptance"]) - set(ACCEPTANCE_ORDER)
    ):
        raise ValueError("qualitative pilot evidence is incomplete")
    counts = (
        assessments.groupby(["condition_id", "acceptance"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=CONDITIONS, columns=ACCEPTANCE_ORDER, fill_value=0)
    )
    if not counts.sum(axis=1).eq(2).all():
        raise ValueError("qualitative pilot evidence must contain two cases per condition")
    return paired, counts


def _interval_panel(axis: plt.Axes, evidence: pd.DataFrame, metric: str) -> None:
    rows = evidence[evidence["metric"].eq(metric)].set_index("condition_id").loc[list(CONDITIONS)]
    y = np.arange(len(CONDITIONS))
    mean = rows["mean_delta"].to_numpy()
    low = rows["ci95_low"].to_numpy()
    high = rows["ci95_high"].to_numpy()
    axis.errorbar(
        mean,
        y,
        xerr=np.vstack([mean - low, high - mean]),
        fmt="o",
        color="#2F6690",
        ecolor="#2F6690",
        markerfacecolor="white",
        markeredgewidth=1.5,
        markersize=5.5,
        linewidth=1.5,
        capsize=3,
    )
    axis.axvline(0, color="#343A40", linewidth=0.9)
    axis.set_yticks(y, [CONDITION_LABELS[item] for item in CONDITIONS])
    axis.grid(axis="x", color="#D9DEE3", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.xaxis.set_major_formatter(FuncFormatter(_comma))
    if metric == "psnr_y":
        axis.set_title("Diferencia de PSNR-Y")
        axis.set_xlabel("EDSR adaptado menos oficial (dB)")
    else:
        axis.set_title("Diferencia de SSIM-Y")
        axis.set_xlabel("EDSR adaptado menos oficial")
        axis.tick_params(axis="y", labelleft=False)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def generate(artifact_root: Path, output: Path) -> None:
    paired, counts = load_evidence(artifact_root)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "#202428",
            "axes.labelcolor": "#202428",
            "axes.edgecolor": "#6C757D",
            "xtick.color": "#343A40",
            "ytick.color": "#343A40",
            "pdf.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(7.2, 7.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 0.9))
    psnr_axis = figure.add_subplot(grid[0, 0])
    ssim_axis = figure.add_subplot(grid[0, 1], sharey=psnr_axis)
    acceptance_axis = figure.add_subplot(grid[1, :])
    _interval_panel(psnr_axis, paired, "psnr_y")
    _interval_panel(ssim_axis, paired, "ssim_y")
    psnr_axis.invert_yaxis()

    y = np.arange(len(CONDITIONS))
    left = np.zeros(len(CONDITIONS), dtype=float)
    for decision in ACCEPTANCE_ORDER:
        values = counts[decision].to_numpy(dtype=float)
        bars = acceptance_axis.barh(
            y,
            values,
            left=left,
            color=ACCEPTANCE_COLORS[decision],
            edgecolor="#343A40",
            linewidth=0.6,
            hatch=ACCEPTANCE_HATCHES[decision],
            label=ACCEPTANCE_LABELS[decision],
        )
        for bar, value in zip(bars, values, strict=True):
            if value:
                acceptance_axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(value)),
                    ha="center",
                    va="center",
                    color="white" if decision != "acceptable-with-reservations" else "#202428",
                    fontweight="bold",
                )
        left += values
    acceptance_axis.set_yticks(y, [CONDITION_LABELS[item] for item in CONDITIONS])
    acceptance_axis.invert_yaxis()
    acceptance_axis.set_xlim(0, 2)
    acceptance_axis.set_xticks([0, 1, 2])
    acceptance_axis.set_xlabel("Casos revisados (dos por condición)")
    acceptance_axis.set_title("Aceptación como derivado de consulta", y=1.14)
    acceptance_axis.grid(axis="x", color="#D9DEE3", linewidth=0.7)
    acceptance_axis.set_axisbelow(True)
    acceptance_axis.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    for side in ("top", "right"):
        acceptance_axis.spines[side].set_visible(False)

    figure.suptitle(
        "Transferencia del EDSR adaptado al corpus externo",
        fontsize=13,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/professional-pilot-v1"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.artifact_root.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
