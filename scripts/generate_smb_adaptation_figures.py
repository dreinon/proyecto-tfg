"""Generate thesis-ready figures for the validated SMB EDSR adaptation study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDITIONS = (
    ("x2-clean", "x2 limpia"),
    ("x2-moderate", "x2 moderada"),
    ("x2-strong", "x2 fuerte"),
    ("x4-clean", "x4 limpia"),
    ("x4-moderate", "x4 moderada"),
    ("x4-strong", "x4 fuerte"),
)
BLUE = "#15558D"
ORANGE = "#B55416"
INK = "#27313A"
GRID = "#D7DCE2"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
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
    for suffix, options in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 220})):
        path = root / f"{stem}{suffix}"
        figure.savefig(path, bbox_inches="tight", **options)
        paths.append(path)
    plt.close(figure)
    return paths


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def adaptation_delta_figure(paired: pd.DataFrame, output_root: Path) -> list[Path]:
    """Show paired adapted-minus-pretrained gains and their source bootstrap intervals."""

    _style()
    comparison = paired[paired["comparator_id"] == "edsr-baseline-official-v1"]
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 7.0))
    figure.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.16, hspace=0.16)
    x = np.arange(len(CONDITIONS))
    for axis, metric, ylabel in zip(
        axes,
        ("psnr_y", "ssim_y"),
        ("Diferencia de PSNR-Y (dB)", "Diferencia de SSIM-Y"),
        strict=True,
    ):
        rows = comparison[comparison["metric"] == metric].set_index("condition_id")
        means = np.array([rows.loc[condition, "mean_delta"] for condition, _ in CONDITIONS])
        lows = np.array([rows.loc[condition, "ci95_low"] for condition, _ in CONDITIONS])
        highs = np.array([rows.loc[condition, "ci95_high"] for condition, _ in CONDITIONS])
        axis.errorbar(
            x,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            fmt="o",
            color=BLUE,
            ecolor=BLUE,
            markerfacecolor="white",
            markeredgecolor=INK,
            markeredgewidth=0.8,
            markersize=6,
            linewidth=1.3,
            capsize=3,
        )
        axis.axhline(0, color=INK, linewidth=0.9, linestyle="--")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [label for _, label in CONDITIONS])
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Ganancia del EDSR adaptado frente al EDSR oficial")
    axes[1].set_xlabel("Condición de degradación")
    figure.text(
        0.01,
        0.025,
        "Puntos: diferencia media pareada. Barras: IC percentil del 95 %, "
        "2.000 remuestreos de 20 obras test independientes.",
        fontsize=8,
        color="#4B5563",
    )
    return _save_all(figure, output_root, "adaptation-deltas-vs-pretrained")


def checkpoint_selection_figure(artifact_root: Path, output_root: Path) -> list[Path]:
    """Show validation-only checkpoint selection for x2 and x4."""

    _style()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    figure.subplots_adjust(left=0.08, right=0.99, top=0.85, bottom=0.23, wspace=0.25)
    colors = {2: BLUE, 4: ORANGE}
    for axis, scale in zip(axes, (2, 4), strict=True):
        history = pd.read_csv(artifact_root / "training" / f"training-history-x{scale}.csv")
        selected = history.loc[history["validation_l1"].idxmin()]
        axis.plot(
            history["step"],
            history["validation_l1"],
            color=colors[scale],
            marker="o",
            markersize=4,
            linewidth=1.5,
        )
        axis.axvline(
            selected["step"],
            color=INK,
            linestyle="--",
            linewidth=1.0,
            label=f"Selección: paso {int(selected['step'])}",
        )
        axis.scatter(
            [selected["step"]],
            [selected["validation_l1"]],
            s=42,
            color="white",
            edgecolor=INK,
            linewidth=0.9,
            zorder=3,
        )
        axis.set_title(f"EDSR x{scale}")
        axis.set_xlabel("Paso de optimización")
        axis.set_ylabel("L1 RGB media de validación")
        axis.set_xlim(-50, 2550)
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="upper right", fontsize=8)
    figure.suptitle("Selección de checkpoints sin consultar test", fontweight="bold")
    figure.text(
        0.01,
        0.03,
        "Validación sobre 13 obras independientes y casos fijados por perfil. "
        "Presupuesto máximo: 2.500 pasos por escala.",
        fontsize=8,
        color="#4B5563",
    )
    return _save_all(figure, output_root, "adaptation-checkpoint-selection")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/kaggle/smb-edsr-finetuning-v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/kaggle/smb-edsr-finetuning-v1-figures"),
    )
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    paired_path = artifact_root / "evaluation/paired-bootstrap.csv"
    paired = pd.read_csv(paired_path)
    paths = [
        *adaptation_delta_figure(paired, output_root),
        *checkpoint_selection_figure(artifact_root, output_root),
    ]
    manifest = {
        "schema_version": 1,
        "record_type": "smb-edsr-finetuning-thesis-figures",
        "inputs": {
            "paired_bootstrap_sha256": _sha256(paired_path),
            "training_history_x2_sha256": _sha256(
                artifact_root / "training/training-history-x2.csv"
            ),
            "training_history_x4_sha256": _sha256(
                artifact_root / "training/training-history-x4.csv"
            ),
        },
        "figures": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in paths
        },
    }
    (output_root / "figure-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(output_root), "files": len(paths)}, indent=2))


if __name__ == "__main__":
    main()
