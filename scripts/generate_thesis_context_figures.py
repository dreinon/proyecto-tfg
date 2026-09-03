"""Generate thesis figures for SMB coverage, degradation scale, and effort."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_MANIFEST = Path(
    "data/manifests/recovery/canonical-pixel-v2/"
    "c8d62c14724b38e1c4a415db503a689dad87f0ce9308cf0121a3fe9bd688f69f/"
    "manifest-records.jsonl.gz"
)

BLUE = "#15558D"
ORANGE = "#B55416"
GREEN = "#557A35"
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


def read_manifest(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            rows.append(
                {
                    "source_group_id": record["source_group_id"],
                    "width": record["image"]["decoded_width"],
                    "height": record["image"]["decoded_height"],
                    "region_count": record["annotations"]["region_count"],
                    "paired_eligible": record["paired_eligible"],
                    "ineligibility_reason": record.get("paired_ineligibility_reason"),
                }
            )
    return pd.DataFrame(rows)


def dataset_profile_figure(records: pd.DataFrame, output_root: Path) -> list[Path]:
    _style()
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.5))
    figure.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.22, wspace=0.34)

    group_sizes = records.groupby("source_group_id").size()
    distribution = Counter(group_sizes.tolist())
    x_values = np.arange(1, max(distribution) + 1)
    axes[0].bar(
        x_values,
        [distribution.get(value, 0) for value in x_values],
        color=BLUE,
        edgecolor=INK,
        linewidth=0.45,
    )
    axes[0].set_title("Páginas por obra fuente")
    axes[0].set_xlabel("Páginas en el grupo")
    axes[0].set_ylabel("Número de obras")
    axes[0].set_xticks([1, 3, 5, 7, 9, 11, 14])

    axes[1].scatter(
        records["width"],
        records["height"],
        s=12,
        alpha=0.42,
        color=ORANGE,
        edgecolors="none",
    )
    axes[1].set_title("Dimensiones de página")
    axes[1].set_xlabel("Anchura (px)")
    axes[1].set_ylabel("Altura (px)")

    region_counts = records["region_count"].value_counts().sort_index()
    axes[2].bar(
        region_counts.index,
        region_counts.values,
        color=GREEN,
        edgecolor=INK,
        linewidth=0.45,
    )
    axes[2].axvline(
        records["region_count"].median(), color=INK, linestyle="--", linewidth=1.0, label="Mediana"
    )
    axes[2].set_title("Regiones anotadas")
    axes[2].set_xlabel("Regiones por página")
    axes[2].set_ylabel("Número de páginas")
    axes[2].legend(loc="upper right")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Perfil estructural del corpus SMB auditado", fontweight="bold")
    figure.text(
        0.01,
        0.03,
        f"{len(records)} páginas, {group_sizes.size} obras fuente. "
        "Cada punto representa una página; no se reproducen imágenes del corpus.",
        fontsize=8,
        color="#4B5563",
    )
    return _save_all(figure, output_root, "smb-dataset-profile")


def staff_scale_figure(sample: pd.DataFrame, output_root: Path) -> list[Path]:
    _style()
    spacing = sample["staff_spacing_px"].astype(float)
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    figure.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.23, wspace=0.28)

    axes[0].hist(spacing, bins=np.arange(6, 20, 1), color=BLUE, edgecolor=INK, linewidth=0.45)
    median = float(spacing.median())
    axes[0].axvline(
        median, color=ORANGE, linestyle="--", linewidth=1.4, label=f"Mediana {median:.2f} px"
    )
    axes[0].set_title("Escala observada en la muestra")
    axes[0].set_xlabel("Separación entre líneas del pentagrama (px)")
    axes[0].set_ylabel("Número de obras")
    axes[0].legend(loc="upper right")

    scale = np.linspace(spacing.min(), spacing.max(), 150)
    axes[1].plot(scale, np.maximum(0.30, 0.05 * scale), color=GREEN, linewidth=2, label="Moderada")
    axes[1].plot(
        scale,
        np.maximum(0.30, 0.15 * scale),
        color=ORANGE,
        linewidth=2,
        linestyle="--",
        label="Fuerte",
    )
    axes[1].fill_between(scale, 0, np.maximum(0.30, 0.05 * scale), color=GREEN, alpha=0.10)
    axes[1].fill_between(scale, 0, np.maximum(0.30, 0.15 * scale), color=ORANGE, alpha=0.08)
    axes[1].set_title("Desenfoque normalizado por pentagrama")
    axes[1].set_xlabel("Separación entre líneas del pentagrama (px)")
    axes[1].set_ylabel(r"Desviación del desenfoque $\sigma$ (px HR)")
    axes[1].legend(loc="upper left")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Cobertura de escala y control de degradación v2", fontweight="bold")
    figure.text(
        0.01,
        0.03,
        f"Muestra de {len(sample)} obras independientes; rango observado "
        f"{spacing.min():.2f}-{spacing.max():.2f} px. La condición limpia no aplica desenfoque.",
        fontsize=8,
        color="#4B5563",
    )
    return _save_all(figure, output_root, "staff-scale-and-blur")


def effort_figure(effort: pd.DataFrame, output_root: Path) -> list[Path]:
    _style()
    # Type 3 avoids a spurious .notdef reference produced by the Matplotlib TrueType subset when
    # this figure is embedded with pdfTeX; veraPDF otherwise rejects the final PDF/A-2b document.
    plt.rcParams["pdf.fonttype"] = 3
    completed = effort[effort["entry_id"] != "EFF-P5-REMAINING"].copy()
    planned = np.array([60, 65, 75, 75, 55], dtype=float)
    reconstructed = completed["estimate_hours"].to_numpy(dtype=float)
    low = completed["low_hours"].to_numpy(dtype=float)
    high = completed["high_hours"].to_numpy(dtype=float)
    phases = ["P1", "P2", "P3", "P4", "P5"]
    x = np.arange(len(phases))
    width = 0.34

    figure, axis = plt.subplots(figsize=(8.6, 4.3))
    figure.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.22)
    axis.bar(
        x - width / 2,
        planned,
        width,
        color=BLUE,
        edgecolor=INK,
        linewidth=0.45,
        label="Plan inicial",
    )
    axis.bar(
        x + width / 2,
        reconstructed,
        width,
        yerr=np.vstack((reconstructed - low, high - reconstructed)),
        capsize=3,
        color=ORANGE,
        edgecolor=INK,
        linewidth=0.45,
        label="Reconstrucción central",
    )
    axis.text(
        0.98,
        0.95,
        "Total reconstruido: 346 h",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=INK,
    )
    axis.set_title("Esfuerzo inicial y reconstruido hasta el 1 de septiembre")
    axis.set_xlabel("Fase")
    axis.set_ylabel("Horas")
    axis.set_xticks(x, phases)
    axis.set_ylim(0, 100)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", visible=False)
    axis.legend(ncols=2, loc="upper left")
    figure.text(
        0.01,
        0.03,
        "Plan: 330 h. Reconstrucción hasta el 1 de septiembre: 346 h "
        "(intervalo plausible 312-380 h). Previsión final tras cierre y piloto: 378-384 h. "
        "Las estimaciones no son un registro horario.",
        fontsize=8,
        color="#4B5563",
    )
    return _save_all(figure, output_root, "effort-plan-vs-reconstruction")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--sample", type=Path, default=Path("data/audits/smb-evaluation-sample-v2.csv")
    )
    parser.add_argument("--effort", type=Path, default=Path("docs/effort-log.csv"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/thesis/context-figures")
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = read_manifest(args.manifest)
    sample = pd.read_csv(args.sample)
    effort = pd.read_csv(args.effort)
    paths = [
        *dataset_profile_figure(records, output_root),
        *staff_scale_figure(sample, output_root),
        *effort_figure(effort, output_root),
    ]
    manifest = {
        "schema_version": 1,
        "record_type": "thesis-context-figures",
        "inputs": {
            "manifest": str(args.manifest),
            "sample": str(args.sample),
            "effort": str(args.effort),
        },
        "figures": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in paths
        },
    }
    (output_root / "figure-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_root": str(output_root), "files": len(paths)}, indent=2))


if __name__ == "__main__":
    main()
