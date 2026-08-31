"""Generate thesis-ready static figures from the final SMB v2 analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = (
    ("bicubic-opencv-v1", "Bicúbica", "#15558D", "//"),
    ("edsr-baseline-official-v1", "EDSR", "#B55416", "xx"),
    ("swinir-lightweight-official-v1", "SwinIR", "#557A35", ".."),
)
CONDITIONS = (
    ("x2-clean", "x2 limpia"),
    ("x2-moderate", "x2 moderada"),
    ("x2-strong", "x2 fuerte"),
    ("x4-clean", "x4 limpia"),
    ("x4-moderate", "x4 moderada"),
    ("x4-strong", "x4 fuerte"),
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D7DCE2",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_all(figure: plt.Figure, root: Path, stem: str) -> list[Path]:
    paths = []
    for suffix, options in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 220}),
    ):
        path = root / f"{stem}{suffix}"
        figure.savefig(path, bbox_inches="tight", **options)
        paths.append(path)
    plt.close(figure)
    return paths


def fidelity_figure(aggregate: pd.DataFrame, output_root: Path) -> list[Path]:
    _style()
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 7.2))
    figure.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.15, hspace=0.12)
    x = np.arange(len(CONDITIONS))
    width = 0.24
    for method_index, (method_id, label, color, hatch) in enumerate(METHODS):
        rows = aggregate[aggregate["method_id"] == method_id].set_index("condition_id")
        offset = (method_index - 1) * width
        for axis, metric in zip(axes, ("psnr_y_mean", "ssim_y_mean"), strict=True):
            values = [rows.loc[condition_id, metric] for condition_id, _ in CONDITIONS]
            axis.bar(
                x + offset,
                values,
                width,
                label=label,
                color=color,
                edgecolor="#27313A",
                linewidth=0.45,
                hatch=hatch,
            )
    axes[0].set_title("Fidelidad media por condición y método")
    axes[0].set_ylabel("PSNR-Y (dB)")
    axes[0].set_ylim(0, 30)
    axes[1].set_ylabel("SSIM-Y")
    axes[1].set_ylim(0.58, 1.0)
    axes[1].set_xlabel("Condición de degradación")
    for axis in axes:
        axis.set_xticks(x, [label for _, label in CONDITIONS])
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncols=3, loc="upper right")
    figure.text(
        0.01,
        0.02,
        "Media de 64 obras independientes por condición. "
        "Comparar métodos solo dentro de cada condición.",
        fontsize=8,
        color="#4B5563",
    )
    return _save_all(figure, output_root, "fidelity-means")


def paired_delta_figure(paired: pd.DataFrame, output_root: Path) -> list[Path]:
    _style()
    baseline = paired[paired["comparator_id"] == "bicubic-opencv-v1"]
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 7.2))
    figure.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.15, hspace=0.12)
    x = np.arange(len(CONDITIONS))
    learned = METHODS[1:]
    offsets = (-0.10, 0.10)
    for axis, metric, ylabel in zip(
        axes,
        ("psnr_y", "ssim_y"),
        ("Diferencia de PSNR-Y (dB)", "Diferencia de SSIM-Y"),
        strict=True,
    ):
        metric_rows = baseline[baseline["metric"] == metric]
        for offset, (method_id, label, color, _) in zip(offsets, learned, strict=True):
            rows = metric_rows[metric_rows["method_id"] == method_id].set_index("condition_id")
            means = np.array(
                [rows.loc[condition_id, "mean_delta"] for condition_id, _ in CONDITIONS]
            )
            lows = np.array([rows.loc[condition_id, "ci95_low"] for condition_id, _ in CONDITIONS])
            highs = np.array(
                [rows.loc[condition_id, "ci95_high"] for condition_id, _ in CONDITIONS]
            )
            axis.errorbar(
                x + offset,
                means,
                yerr=np.vstack((means - lows, highs - means)),
                fmt="o",
                markersize=5,
                capsize=3,
                color=color,
                markeredgecolor="#27313A",
                markeredgewidth=0.5,
                linewidth=1.2,
                label=label,
            )
        axis.axhline(0, color="#27313A", linewidth=0.9, linestyle="--")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [label for _, label in CONDITIONS])
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Diferencias pareadas frente a interpolación bicúbica")
    axes[0].legend(ncols=2, loc="upper right")
    axes[1].set_xlabel("Condición de degradación")
    figure.text(
        0.01,
        0.02,
        "Puntos: diferencia media. Barras: IC percentil del 95 %, 2.000 remuestreos de 64 obras.",
        fontsize=8,
        color="#4B5563",
    )
    return _save_all(figure, output_root, "paired-deltas-vs-bicubic")


def runtime_figure(runtime: pd.DataFrame, output_root: Path) -> list[Path]:
    _style()
    figure, axis = plt.subplots(figsize=(8.5, 4.4))
    figure.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.22)
    x = np.arange(len(CONDITIONS))
    width = 0.24
    for method_index, (method_id, label, color, hatch) in enumerate(METHODS):
        rows = runtime[runtime["method_id"] == method_id].set_index("condition_id")
        values = [
            rows.loc[condition_id, "runtime_seconds_median"] for condition_id, _ in CONDITIONS
        ]
        axis.bar(
            x + (method_index - 1) * width,
            values,
            width,
            label=label,
            color=color,
            edgecolor="#27313A",
            linewidth=0.45,
            hatch=hatch,
        )
    axis.set_title("Tiempo mediano de inferencia por página")
    axis.set_ylabel("Segundos por página (escala logarítmica)")
    axis.set_xlabel("Condición de degradación")
    axis.set_yscale("log")
    axis.set_xticks(x, [label for _, label in CONDITIONS])
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncols=3, loc="upper right")
    figure.text(
        0.01,
        0.03,
        "Ejecución final en Kaggle con dos Tesla T4; tiempos no portables a otro hardware.",
        fontsize=8,
        color="#4B5563",
    )
    return _save_all(figure, output_root, "runtime-median")


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("artifacts/kaggle/phase3-smb-analysis-v2/final"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/kaggle/phase3-smb-analysis-v2/final/figures"),
    )
    args = parser.parse_args()
    analysis_root = args.analysis_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    aggregate = pd.read_csv(analysis_root / "aggregate-metrics.csv")
    paired = pd.read_csv(analysis_root / "paired-bootstrap.csv")
    runtime = pd.read_csv(analysis_root / "runtime-summary.csv")
    paths = [
        *fidelity_figure(aggregate, output_root),
        *paired_delta_figure(paired, output_root),
        *runtime_figure(runtime, output_root),
    ]
    manifest = {
        "schema_version": 1,
        "record_type": "smb-v2-thesis-figures",
        "figures": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in paths
        },
    }
    manifest_path = output_root / "figure-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_root": str(output_root), "files": len(paths)}, indent=2))


if __name__ == "__main__":
    main()
