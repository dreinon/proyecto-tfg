"""Generate reproducible qualitative figures for the thesis from retained evidence.

The script never selects cases from model scores. It uses the qualitative cases that were
fixed before review and regenerates only an explanatory degradation sequence from the frozen
staff-scale protocol. It produces both detail crops and a bounded set of complete-page comparison
plates needed to expose global document behaviour and the retained external failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from score_super_resolution.staff_scale import (
    apply_scale_normalized_degradation,
    load_scale_normalized_control,
)

INK = "#202428"
ORANGE = "#B55416"
BORDER = "#7B8791"


@dataclass(frozen=True)
class FigureRecord:
    """One generated figure and its scientific lineage."""

    filename: str
    inputs: tuple[str, ...]
    crop_xyxy: tuple[int, int, int, int] | None
    note: str
    detail_regions: tuple[dict, ...] = ()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "text.color": INK,
            "axes.titlecolor": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _crop(pixels: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    if not (0 <= left < right <= pixels.shape[1] and 0 <= top < bottom <= pixels.shape[0]):
        raise ValueError(f"crop {box} falls outside image {pixels.shape[1]}x{pixels.shape[0]}")
    return np.ascontiguousarray(pixels[top:bottom, left:right])


def _image_axis(axis: plt.Axes, pixels: np.ndarray, title: str, *, title_size: float = 9.0) -> None:
    axis.imshow(pixels, interpolation="nearest")
    axis.set_title(title, pad=5, fontweight="bold", fontsize=title_size)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(BORDER)
        spine.set_linewidth(0.7)


def _save(figure: plt.Figure, output_root: Path, filename: str) -> Path:
    output = output_root / filename
    figure.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        pil_kwargs={"compress_level": 9},
    )
    plt.close(figure)
    return output


def _score_anatomy(adaptation_root: Path, output_root: Path) -> FigureRecord:
    source = adaptation_root / "smb-test-000451/x4-moderate/reference-hr.png"
    hr = _read_rgb(source)
    overview_box = (100, 320, 2440, 920)
    details = (
        ((160, 340, 680, 850), "Clave, compás y dinámica"),
        ((690, 340, 1450, 820), "Cabezas, plicas y ligaduras"),
        ((1450, 340, 2380, 870), "Alteraciones y acordes"),
    )
    overview = _crop(hr, overview_box)

    figure = plt.figure(figsize=(7.25, 4.25), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.15, 1.0))
    top = figure.add_subplot(grid[0, :])
    _image_axis(top, overview, "Un sistema musical como imagen estructurada")
    for index, (box, _label) in enumerate(details, start=1):
        left, upper, right, lower = box
        rectangle = Rectangle(
            (left - overview_box[0], upper - overview_box[1]),
            right - left,
            lower - upper,
            fill=False,
            edgecolor=ORANGE,
            linewidth=1.4,
        )
        top.add_patch(rectangle)
        top.text(
            left - overview_box[0] + 8,
            upper - overview_box[1] + 18,
            str(index),
            color="white",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "circle,pad=0.18", "facecolor": ORANGE, "edgecolor": ORANGE},
        )
    for index, (box, label) in enumerate(details):
        axis = figure.add_subplot(grid[1, index])
        _image_axis(axis, _crop(hr, box), f"{index + 1}. {label}", title_size=7.4)

    _save(figure, output_root, "score-anatomy.png")
    return FigureRecord(
        filename="score-anatomy.png",
        inputs=(str(source),),
        crop_xyxy=overview_box,
        note="Explanatory crop from the predeclared x4-moderate adaptation case.",
    )


def _degradation_profiles(
    project_root: Path, adaptation_root: Path, output_root: Path
) -> FigureRecord:
    case_root = adaptation_root / "smb-test-000451/x4-moderate"
    source = case_root / "reference-hr.png"
    hr = _read_rgb(source)
    control = load_scale_normalized_control(project_root)
    spacing = 20.490497
    condition_ids = ("x4-clean", "x4-moderate", "x4-strong")
    inputs: list[np.ndarray] = []
    trace_ids: list[str] = []
    for condition_id in condition_ids:
        result = apply_scale_normalized_degradation(
            hr,
            control=control,
            condition_id=condition_id,
            item_id="smb-test-000451",
            source_group_id="joplin_newrag",
            staff_spacing_px=spacing,
        )
        nearest = cv2.resize(
            result.pixels,
            (hr.shape[1], hr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        inputs.append(nearest)
        trace_ids.append(str(result.trace["trace_id"]))

    retained_moderate = _read_rgb(case_root / "input-lr-nearest.png")
    if not np.array_equal(inputs[1], retained_moderate):
        raise RuntimeError("regenerated x4-moderate input differs from retained Kaggle evidence")

    crop_box = (1320, 340, 2380, 900)
    panels = (hr, *inputs)
    titles = (
        "Referencia HR",
        "LR limpia\nsolo reducción",
        "LR moderada\n+ desenfoque, ruido y JPEG",
        "LR fuerte\n+ desenfoque, ruido y JPEG",
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.25, 4.35), constrained_layout=True)
    for axis, pixels, title in zip(axes.flat, panels, titles, strict=True):
        _image_axis(axis, _crop(pixels, crop_box), title, title_size=8.2)
    _save(figure, output_root, "degradation-profiles-x4.png")
    return FigureRecord(
        filename="degradation-profiles-x4.png",
        inputs=(
            str(source),
            str(case_root / "input-lr-nearest.png"),
            str(project_root / "configs/degradations/staff-scale-score-v2.yaml"),
        ),
        crop_xyxy=crop_box,
        note="Frozen v2 x4 profiles; trace_ids=" + ",".join(trace_ids),
    )


def _comparison_figure(
    *,
    case_root: Path,
    filenames: tuple[str, ...],
    titles: tuple[str, ...],
    crop_box: tuple[int, int, int, int],
    output_root: Path,
    output_name: str,
    note: str,
) -> FigureRecord:
    paths = tuple(case_root / filename for filename in filenames)
    images = tuple(_read_rgb(path) for path in paths)
    shapes = {image.shape for image in images}
    if len(shapes) != 1:
        raise ValueError(f"unaligned evidence in {case_root}")
    figure = plt.figure(figsize=(7.25, 4.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 6)
    if len(images) == 4:
        axes = (
            figure.add_subplot(grid[0, 0:3]),
            figure.add_subplot(grid[0, 3:6]),
            figure.add_subplot(grid[1, 0:3]),
            figure.add_subplot(grid[1, 3:6]),
        )
    elif len(images) == 5:
        axes = (
            figure.add_subplot(grid[0, 0:2]),
            figure.add_subplot(grid[0, 2:4]),
            figure.add_subplot(grid[0, 4:6]),
            figure.add_subplot(grid[1, 1:3]),
            figure.add_subplot(grid[1, 3:5]),
        )
    else:
        raise ValueError("comparison figures support four or five aligned panels")
    for axis, pixels, title in zip(axes, images, titles, strict=True):
        _image_axis(axis, _crop(pixels, crop_box), title)
    _save(figure, output_root, output_name)
    return FigureRecord(
        filename=output_name,
        inputs=tuple(str(path) for path in paths),
        crop_xyxy=crop_box,
        note=note,
    )


def _full_page_comparison(
    *,
    case_root: Path,
    output_root: Path,
    output_name: str,
    note: str,
) -> FigureRecord:
    filenames = (
        "reference-hr.png",
        "input-lr-nearest.png",
        "edsr-baseline-official-v1.png",
        "edsr-smb-finetuned-v1.png",
    )
    titles = ("Referencia HR", "Entrada LR", "EDSR oficial", "EDSR adaptado")
    paths = tuple(case_root / filename for filename in filenames)
    images = tuple(_read_rgb(path) for path in paths)
    shapes = {image.shape for image in images}
    if len(shapes) != 1:
        raise ValueError(f"unaligned full-page evidence in {case_root}")

    height, width = images[0].shape[:2]
    figure_height = 8.15 if height >= width else 5.15
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.25, figure_height),
        constrained_layout=True,
    )
    for axis, pixels, title in zip(axes.flat, images, titles, strict=True):
        _image_axis(axis, pixels, title, title_size=8.2)
    _save(figure, output_root, output_name)
    return FigureRecord(
        filename=output_name,
        inputs=tuple(str(path) for path in paths),
        crop_xyxy=None,
        note=note,
    )


def _detail_gallery(
    *,
    project_root: Path,
    rows: tuple[tuple[Path, tuple[int, int, int, int], str], ...],
    filenames: tuple[str, ...],
    titles: tuple[str, ...],
    output_root: Path,
    output_name: str,
) -> FigureRecord:
    """Compare unchanged pixels in matched detail regions of fixed reviewed cases."""
    figure = plt.figure(figsize=(7.25, 1.95 * len(rows)), constrained_layout=True)
    grid = figure.add_gridspec(len(rows) * 2, len(filenames), height_ratios=[0.17, 1] * len(rows))
    inputs = []
    regions = []
    for index, (case_root, box, label) in enumerate(rows):
        heading = figure.add_subplot(grid[index * 2, :])
        heading.axis("off")
        heading.text(0, 0.5, label, fontsize=8.4, fontweight="bold", va="center")
        paths = tuple(case_root / name for name in filenames)
        images = tuple(_read_rgb(path) for path in paths)
        if len({pixels.shape for pixels in images}) != 1:
            raise ValueError(f"unaligned detail evidence in {case_root}")
        for column, (pixels, title) in enumerate(zip(images, titles, strict=True)):
            axis = figure.add_subplot(grid[index * 2 + 1, column])
            _image_axis(axis, _crop(pixels, box), title, title_size=7.7)
        inputs.extend(str(path) for path in paths)
        regions.append(
            {
                "case_path": str(case_root.relative_to(project_root)),
                "crop_xyxy": list(box),
                "label": label,
            }
        )
    _save(figure, output_root, output_name)
    return FigureRecord(
        filename=output_name,
        inputs=tuple(dict.fromkeys(inputs)),
        crop_xyxy=None,
        note=(
            "Editorial detail regions from already fixed reviewed cases; identical coordinates "
            "within each row, nearest-neighbour display, no enhancement or new assessment."
        ),
        detail_regions=tuple(regions),
    )


def _write_manifest(
    output_root: Path, project_root: Path, records: tuple[FigureRecord, ...]
) -> None:
    payload = {
        "schema_version": 1,
        "record_type": "thesis-qualitative-figures",
        "selection_policy": "predeclared-qualitative-cases-only",
        "project_revision": _git_revision(project_root),
        "generator": {
            "path": str(Path(__file__).resolve().relative_to(project_root)),
            "sha256": _sha256(Path(__file__)),
        },
        "publication_basis": {
            "path": "docs/professional-pilot-publication-basis.md",
            "sha256": _sha256(project_root / "docs/professional-pilot-publication-basis.md"),
        },
        "figures": [],
    }
    for record in records:
        output = output_root / record.filename
        payload["figures"].append(
            {
                "filename": record.filename,
                "sha256": _sha256(output),
                "bytes": output.stat().st_size,
                "inputs": [
                    {
                        "path": str(Path(path).resolve().relative_to(project_root)),
                        "sha256": _sha256(Path(path)),
                    }
                    for path in record.inputs
                ],
                "display_scope": (
                    "analytical-crop"
                    if record.crop_xyxy is not None or record.detail_regions
                    else "full-page"
                ),
                "crop_xyxy": list(record.crop_xyxy) if record.crop_xyxy is not None else None,
                **(
                    {"detail_regions": list(record.detail_regions)} if record.detail_regions else {}
                ),
                "note": record.note,
            }
        )
    (output_root / "qualitative-figure-manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_revision(project_root: Path) -> str:
    head = project_root / ".git/HEAD"
    if not head.is_file():
        return "unavailable"
    value = head.read_text(encoding="utf-8").strip()
    if not value.startswith("ref: "):
        return value
    reference = project_root / ".git" / value.removeprefix("ref: ")
    return reference.read_text(encoding="utf-8").strip() if reference.is_file() else "unavailable"


def generate(project_root: Path, output_root: Path) -> tuple[FigureRecord, ...]:
    _style()
    adaptation_root = (
        project_root / "artifacts/kaggle/smb-edsr-finetuning-v1/evaluation/qualitative"
    )
    pretrained_root = project_root / "artifacts/kaggle/phase3-smb-evaluation-v2/qualitative"
    external_root = project_root / "artifacts/professional-pilot-v1/qualitative"
    external_cases = (
        (
            "cullera-suite",
            "x2-clean",
            "acceptable",
            "external-cullera-suite-full-page-x2-clean.png",
        ),
        (
            "el-jardin-de-hera",
            "x2-clean",
            "acceptable",
            "external-el-jardin-de-hera-full-page-x2-clean.png",
        ),
        (
            "how-to-train-your-dragon",
            "x2-moderate",
            "acceptable",
            "external-how-to-train-your-dragon-full-page-x2-moderate.png",
        ),
        (
            "xativa-1939",
            "x2-moderate",
            "acceptable",
            "external-xativa-1939-full-page-x2-moderate.png",
        ),
        (
            "malaguenya-de-barxeta",
            "x2-strong",
            "acceptable",
            "external-malaguenya-de-barxeta-full-page-x2-strong.png",
        ),
        (
            "three-revelations",
            "x2-strong",
            "acceptable",
            "external-accepted-full-page-x2-strong.png",
        ),
        (
            "capitania-cides",
            "x4-clean",
            "acceptable",
            "external-capitania-cides-full-page-x4-clean.png",
        ),
        (
            "la-rosa-i-el-drac",
            "x4-clean",
            "acceptable-with-reservations",
            "external-la-rosa-i-el-drac-full-page-x4-clean.png",
        ),
        (
            "city-in-three-words",
            "x4-moderate",
            "acceptable-with-reservations",
            "external-city-in-three-words-full-page-x4-moderate.png",
        ),
        (
            "lorencin-mendoza",
            "x4-moderate",
            "acceptable",
            "external-lorencin-mendoza-full-page-x4-moderate.png",
        ),
        (
            "jurassic-park",
            "x4-strong",
            "acceptable-with-reservations",
            "external-jurassic-park-full-page-x4-strong.png",
        ),
        (
            "mari-carmen",
            "x4-strong",
            "rejected",
            "external-rejected-full-page-x4-strong.png",
        ),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    core_records = (
        _score_anatomy(adaptation_root, output_root),
        _degradation_profiles(project_root, adaptation_root, output_root),
        _comparison_figure(
            case_root=pretrained_root / "smb-test-000210/x4-moderate",
            filenames=(
                "reference-hr.png",
                "input-lr-nearest.png",
                "bicubic-opencv-v1.png",
                "edsr-baseline-official-v1.png",
                "swinir-lightweight-official-v1.png",
            ),
            titles=("HR", "LR", "Bicúbica", "EDSR", "SwinIR"),
            crop_box=(610, 60, 1210, 380),
            output_root=output_root,
            output_name="pretrained-comparison-x4-moderate.png",
            note="Predeclared SMB v2 x4-moderate qualitative case.",
        ),
        _comparison_figure(
            case_root=adaptation_root / "smb-test-000451/x4-moderate",
            filenames=(
                "reference-hr.png",
                "input-lr-nearest.png",
                "edsr-baseline-official-v1.png",
                "edsr-smb-finetuned-v1.png",
            ),
            titles=("HR", "LR", "EDSR oficial", "EDSR adaptado"),
            crop_box=(1320, 340, 2380, 900),
            output_root=output_root,
            output_name="adapted-comparison-x4-moderate.png",
            note="Predeclared adaptation x4-moderate qualitative case.",
        ),
        _comparison_figure(
            case_root=adaptation_root / "smb-test-000655/x4-strong",
            filenames=(
                "reference-hr.png",
                "input-lr-nearest.png",
                "edsr-baseline-official-v1.png",
                "edsr-smb-finetuned-v1.png",
            ),
            titles=("HR", "LR", "EDSR oficial", "EDSR adaptado"),
            crop_box=(610, 105, 1190, 420),
            output_root=output_root,
            output_name="adapted-limit-x4-strong.png",
            note="Predeclared adaptation x4-strong qualitative failure case.",
        ),
        _full_page_comparison(
            case_root=adaptation_root / "smb-test-000451/x4-moderate",
            output_root=output_root,
            output_name="adapted-full-page-x4-moderate.png",
            note=(
                "Complete-page view of the predeclared SMB adaptation x4-moderate case; "
                "published for non-commercial academic analysis under the dataset licence."
            ),
        ),
    )
    external_records = tuple(
        _full_page_comparison(
            case_root=external_root / work_id / condition_id,
            output_root=output_root,
            output_name=output_name,
            note=(
                f"Complete-page view of predeclared external case {work_id}/{condition_id}; "
                f"student decision={decision}; reproduced under the bounded SJMA authorization "
                "and academic-analysis record."
            ),
        )
        for work_id, condition_id, decision, output_name in external_cases
    )
    detail_records = (
        _detail_gallery(
            project_root=project_root,
            rows=(
                (
                    pretrained_root / "smb-test-000252/x2-clean",
                    (170, 150, 490, 350),
                    "A · x2 limpia · Notación y separaciones",
                ),
                (
                    pretrained_root / "smb-test-000264/x2-strong",
                    (310, 25, 630, 225),
                    "B · x2 fuerte · Textura y contornos",
                ),
                (
                    pretrained_root / "smb-test-000526/x4-strong",
                    (260, 140, 580, 340),
                    "C · x4 fuerte · Trazos finos y texto",
                ),
            ),
            filenames=(
                "reference-hr.png",
                "input-lr-nearest.png",
                "edsr-baseline-official-v1.png",
                "swinir-lightweight-official-v1.png",
            ),
            titles=("HR", "Entrada LR", "EDSR", "SwinIR"),
            output_root=output_root,
            output_name="pretrained-detail-gallery.png",
        ),
        _detail_gallery(
            project_root=project_root,
            rows=(
                (
                    adaptation_root / "smb-test-000451/x4-moderate",
                    (1590, 460, 1910, 660),
                    "A · x4 moderada · Pentagrama, alteraciones y contornos",
                ),
                (
                    adaptation_root / "smb-test-000655/x4-strong",
                    (650, 120, 970, 320),
                    "B · x4 fuerte · Digitación y símbolos pequeños",
                ),
            ),
            filenames=(
                "reference-hr.png",
                "input-lr-nearest.png",
                "edsr-baseline-official-v1.png",
                "edsr-smb-finetuned-v1.png",
            ),
            titles=("HR", "Entrada LR", "EDSR oficial", "EDSR adaptado"),
            output_root=output_root,
            output_name="adapted-detail-gallery.png",
        ),
        _detail_gallery(
            project_root=project_root,
            rows=(
                (
                    external_root / "three-revelations/x2-strong",
                    (670, 460, 1030, 680),
                    "A · Three Revelations · x2 fuerte · Aceptable para consulta",
                ),
                (
                    external_root / "la-rosa-i-el-drac/x4-clean",
                    (360, 255, 720, 475),
                    "B · La rosa i el drac · x4 limpia · Aceptable con reservas",
                ),
                (
                    external_root / "mari-carmen/x4-strong",
                    (600, 260, 960, 480),
                    "C · Mari Carmen · x4 fuerte · Rechazado",
                ),
            ),
            filenames=(
                "reference-hr.png",
                "input-lr-nearest.png",
                "edsr-baseline-official-v1.png",
                "edsr-smb-finetuned-v1.png",
            ),
            titles=("HR", "Entrada LR", "EDSR oficial", "EDSR adaptado"),
            output_root=output_root,
            output_name="external-detail-gallery.png",
        ),
    )
    records = (*core_records, *external_records, *detail_records)
    _write_manifest(output_root, project_root, records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/thesis/qualitative-figures"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root.is_absolute()
        else (project_root / args.output_root).resolve()
    )
    records = generate(project_root, output_root)
    print(json.dumps({"output_root": str(output_root), "figures": len(records)}, indent=2))


if __name__ == "__main__":
    main()
