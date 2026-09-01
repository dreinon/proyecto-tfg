"""Build the reader-facing, locally preflightable Kaggle adaptation notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


def markdown(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source.strip())


def build_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook.cells = [
        markdown(
            """
# Adaptación acotada de EDSR a partituras SMB

## En resumen

Este estudio secundario comprueba si un ajuste fino específico del dominio mejora EDSR frente a
su checkpoint oficial y frente a interpolación bicúbica. La unidad de separación es la obra, no la
página ni el parche: 45 obras previamente usadas solo para desarrollo forman *train*; 13 obras
nuevas forman validación; 20 obras nuevas forman el test final. Las 64 obras del benchmark
principal v2 están excluidas de las tres particiones.

El notebook no abre el test hasta haber seleccionado los checkpoints x2 y x4 mediante la pérdida
L1 de validación. Los resultados y conclusiones aparecerán únicamente tras una ejecución completa
en Kaggle; una ejecución local valida el contrato sin entrenar ni inspeccionar el test.
"""
        ),
        markdown(
            """
## Contexto y método

- Datos: revisión inmutable de `PRAIG/SMB`, con identidad de píxeles contrastada contra el
  manifiesto auditado del proyecto.
- Tarea: super-resolución x2 y x4 bajo los perfiles `clean`, `moderate` y `strong` del protocolo
  normalizado por escala de pentagrama v2.
- Ajuste: EDSR oficial, pérdida RGB-L1, parches centrados en regiones de notación, muestreo
  equilibrado por obra y por severidad.
- Selección: menor L1 media sobre una página representativa prefijada de cada obra de validación.
- Test: una página representativa prefijada por cada una de 20 obras nuevas; comparación pareada
  por obra mediante PSNR-Y, SSIM-Y e intervalos bootstrap.
- Lectura cualitativa: seis casos fijados antes del entrenamiento, mostrando HR, LR ampliada,
  bicúbica, EDSR oficial y EDSR ajustado. Una mejora métrica no se interpretará automáticamente
  como ausencia de errores musicales.
"""
        ),
        markdown(
            """
## Preparación del entorno

En Kaggle activa Internet, una GPU compatible y el secreto `HF_TOKEN`. El montaje conserva el
PyTorch del runtime para no romper su compatibilidad CUDA. Si el repositorio ya existe en
`/kaggle/working/proyecto-tfg`, se reutiliza; en caso contrario se clona desde GitHub.
"""
        ),
        code(
            r"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_URL = "https://github.com/dreinon/proyecto-tfg.git"
IS_KAGGLE = Path("/kaggle/working").is_dir()


def discover_project_root() -> Path:
    starts = [Path.cwd(), Path("/kaggle/working/proyecto-tfg")]
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate.resolve()
    if not IS_KAGGLE:
        raise RuntimeError("No se encuentra el repositorio del proyecto.")
    if shutil.which("uv") is None:
        raise RuntimeError("El runtime de Kaggle no incluye uv.")
    destination = Path("/kaggle/working/proyecto-tfg")
    subprocess.run(["git", "clone", "--depth", "1", REPOSITORY_URL, str(destination)], check=True)
    return destination.resolve()


PROJECT_ROOT = discover_project_root()
if IS_KAGGLE:
    torch_before = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["uv", "pip", "install", "--system", "--no-deps", "-e", str(PROJECT_ROOT)],
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--system",
            "--no-deps",
            "cairosvg==2.9.0",
            "cairocffi==1.7.1",
            "cssselect2==0.9.0",
            "defusedxml==0.7.1",
            "tinycss2==1.5.1",
            "webencodings==0.5.1",
        ],
        check=True,
    )
    torch_after = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if torch_after != torch_before:
        raise RuntimeError("La preparación reemplazó PyTorch; reinicia el runtime.")
    subprocess.run(
        [sys.executable, "-c", "import cairocffi, cssselect2, cairosvg, cv2, datasets, yaml"],
        check=True,
    )

source_path = str(PROJECT_ROOT / "src")
if source_path not in sys.path:
    sys.path.insert(0, source_path)
os.chdir(PROJECT_ROOT)

if IS_KAGGLE:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

git_revision = subprocess.run(
    ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
{"project_root": str(PROJECT_ROOT), "git_revision": git_revision, "is_kaggle": IS_KAGGLE}
"""
        ),
        markdown("## Contrato congelado y particiones"),
        code(
            r"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import FileLink, display

from score_super_resolution.adaptation_split import load_frozen_adaptation_split
from score_super_resolution.edsr_finetuning import (
    ADAPTED_METHOD_ID,
    BICUBIC_METHOD_ID,
    PRETRAINED_METHOD_ID,
    analyze_adaptation_results,
    evaluate_adaptation,
    export_qualitative_cases,
    fine_tune_edsr_scale,
    load_adaptation_dataset,
    load_completed_finetuning_run,
    preflight_adaptation_dataset,
    write_run_manifest,
)

SEED = 20260902
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

FULL_RUN = IS_KAGGLE or os.environ.get("RUN_SMB_EDSR_FINETUNING") == "1"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/smb-edsr-finetuning-v1"
TRAINING_ROOT = ARTIFACT_ROOT / "training"
EVALUATION_ROOT = ARTIFACT_ROOT / "evaluation"
split = load_frozen_adaptation_split(PROJECT_ROOT)
config = split.config
partition_summary = pd.DataFrame(
    [
        {
            "partition": partition,
            "pages": len(split.rows_for(partition)),
            "works": len({row.source_group_id for row in split.rows_for(partition)}),
            "representative_pages": len(split.rows_for(partition, representative_only=True)),
            "prior_role": sorted({row.prior_role for row in split.rows_for(partition)}),
        }
        for partition in ("train", "validation", "test")
    ]
)
display(partition_summary)
{
    "experiment_id": config["experiment_id"],
    "split_sha256": split.split_sha256,
    "source_revision": split.source_revision,
    "full_run": FULL_RUN,
}
"""
        ),
        markdown(
            """
## Verificación de datos de desarrollo

Esta verificación compara la identidad de cada página de train/validación con el manifiesto. No
calcula resultados de SR. El test permanece cerrado en esta etapa.
"""
        ),
        code(
            r"""
dataset = None
development_preflight = None
if not FULL_RUN:
    print("Modo local de contrato: no se carga SMB, no se entrena y no se abre el test.")
else:
    if not torch.cuda.is_available():
        raise RuntimeError("Activa una GPU en Kaggle antes de ejecutar el entrenamiento completo.")
    torch.zeros(1, device="cuda")
    dataset = load_adaptation_dataset(split)
    development_preflight = preflight_adaptation_dataset(
        PROJECT_ROOT,
        dataset,
        split,
        partitions=("train", "validation"),
    )
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "development-preflight.json").write_text(
        json.dumps(asdict(development_preflight), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(asdict(development_preflight))
"""
        ),
        markdown(
            """
## Ajuste fino y selección por validación

Cada escala conserva el checkpoint con menor L1 de validación. Si Kaggle mantiene los archivos y
la celda se repite, un checkpoint completo y compatible se recupera en lugar de entrenarse de
nuevo. Ninguna página del test interviene aquí.
"""
        ),
        code(
            r"""
runs = {}
if dataset is not None:
    for scale in config["methods"]["scales"]:
        checkpoint_path = TRAINING_ROOT / f"edsr-smb-finetuned-v1-x{scale}.pt"
        history_path = TRAINING_ROOT / f"training-history-x{scale}.csv"
        if checkpoint_path.is_file() and history_path.is_file():
            run = load_completed_finetuning_run(
                checkpoint_path,
                history_path,
                split,
                scale=scale,
            )
            print(f"x{scale}: checkpoint compatible recuperado.")
        else:
            run = fine_tune_edsr_scale(
                PROJECT_ROOT,
                dataset,
                split,
                scale=scale,
                output_root=TRAINING_ROOT,
                data_preflight=development_preflight,
            )
        runs[scale] = run

run_summary = pd.DataFrame(
    [
        {
            "scale": scale,
            "best_step": run.best_step,
            "best_validation_l1": run.best_validation_l1,
            "completed_steps": run.completed_steps,
            "stopped_early": run.stopped_early,
            "checkpoint_sha256": run.checkpoint_sha256,
        }
        for scale, run in sorted(runs.items())
    ]
)
run_summary
"""
        ),
        code(
            r"""
if runs:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, (scale, run) in zip(axes, sorted(runs.items()), strict=True):
        history = pd.DataFrame(run.history)
        axis.plot(history.step, history.validation_l1, marker="o", label="validación RGB-L1")
        trained = history.dropna(subset=["training_l1"])
        axis.plot(trained.step, trained.training_l1, marker=".", label="entrenamiento RGB-L1")
        axis.axvline(run.best_step, color="black", linestyle="--", alpha=0.6, label="selección")
        axis.set(title=f"EDSR x{scale}", xlabel="Paso", ylabel="L1 media")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(ARTIFACT_ROOT / "training-curves.png", dpi=220, bbox_inches="tight")
    plt.show()
"""
        ),
        markdown(
            """
## Apertura única del test y evaluación final

La celda exige los dos checkpoints ya seleccionados antes de verificar los píxeles de test. Cada
obra aporta exactamente una página a cada condición. Las comparaciones son pareadas por obra.
"""
        ),
        code(
            r"""
test_preflight = None
results = pd.DataFrame()
if dataset is not None:
    if set(runs) != {2, 4}:
        raise RuntimeError("El test no se abrirá sin los dos checkpoints seleccionados.")
    test_preflight = preflight_adaptation_dataset(
        PROJECT_ROOT,
        dataset,
        split,
        partitions=("test",),
    )
    (ARTIFACT_ROOT / "test-preflight.json").write_text(
        json.dumps(asdict(test_preflight), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_paths = {scale: run.checkpoint_path for scale, run in runs.items()}
    results = evaluate_adaptation(
        PROJECT_ROOT,
        dataset,
        split,
        checkpoint_paths=checkpoint_paths,
        output_root=EVALUATION_ROOT,
        data_preflight=test_preflight,
    )
results.head() if not results.empty else results
"""
        ),
        markdown("## Resultados cuantitativos"),
        code(
            r"""
aggregate = pd.DataFrame()
paired = pd.DataFrame()
if not results.empty:
    aggregate, paired = analyze_adaptation_results(
        results,
        seed=int(config["evaluation"]["bootstrap_seed"]),
        repetitions=int(config["evaluation"]["bootstrap_repetitions"]),
    )
    aggregate.to_csv(EVALUATION_ROOT / "aggregate-metrics.csv", index=False)
    paired.to_csv(EVALUATION_ROOT / "paired-bootstrap.csv", index=False)
display(aggregate)
display(paired)
"""
        ),
        code(
            r"""
if not aggregate.empty:
    conditions = list(config["evaluation"]["conditions"])
    methods = [BICUBIC_METHOD_ID, PRETRAINED_METHOD_ID, ADAPTED_METHOD_ID]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for method in methods:
        frame = aggregate[aggregate.method_id == method].set_index("condition_id").loc[conditions]
        axes[0].plot(conditions, frame.psnr_y_mean, marker="o", label=method)
        axes[1].plot(conditions, frame.ssim_y_mean, marker="o", label=method)
    axes[0].set(title="PSNR-Y medio en test nuevo", xlabel="Condición", ylabel="dB")
    axes[1].set(title="SSIM-Y medio en test nuevo", xlabel="Condición", ylabel="SSIM")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(ARTIFACT_ROOT / "test-metrics.png", dpi=220, bbox_inches="tight")
    plt.show()
"""
        ),
        markdown(
            """
## Casos cualitativos prefijados

Se exportan cinco imágenes por caso: referencia HR, observación LR ampliada sin suavizado,
bicúbica, EDSR oficial y EDSR ajustado. Los hashes de las tres reconstrucciones deben coincidir
con los registrados durante la evaluación cuantitativa.
"""
        ),
        code(
            r"""
qualitative_index = pd.DataFrame()
if not results.empty:
    qualitative_index = export_qualitative_cases(
        PROJECT_ROOT,
        dataset,
        split,
        results,
        checkpoint_paths={scale: run.checkpoint_path for scale, run in runs.items()},
        output_root=EVALUATION_ROOT,
        data_preflight=test_preflight,
    )
qualitative_index.head(10) if not qualitative_index.empty else qualitative_index
"""
        ),
        code(
            r"""
if not qualitative_index.empty:
    labels = [
        "reference-hr",
        "input-lr-nearest",
        BICUBIC_METHOD_ID,
        PRETRAINED_METHOD_ID,
        ADAPTED_METHOD_ID,
    ]
    for (item_id, condition_id), case in qualitative_index.groupby(["item_id", "condition_id"]):
        figure, axes = plt.subplots(1, 5, figsize=(20, 5))
        for axis, label in zip(axes, labels, strict=True):
            path = EVALUATION_ROOT / case.set_index("image_role").loc[label, "path"]
            pixels = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            height, width = pixels.shape[:2]
            crop_height, crop_width = min(640, height), min(1200, width)
            top = max(0, (height - crop_height) // 2)
            left = max(0, (width - crop_width) // 2)
            axis.imshow(pixels[top : top + crop_height, left : left + crop_width])
            axis.set_title(label, fontsize=8)
            axis.axis("off")
        figure.suptitle(f"{item_id} · {condition_id}")
        figure.tight_layout()
        plt.show()
"""
        ),
        markdown(
            """
## Reconciliación, evidencia y descarga

No se formula una conclusión si falta una tupla, un caso cualitativo, un checkpoint o una
verificación de identidad. El ZIP incluye resultados, checkpoints, entorno, entradas congeladas
y un manifiesto SHA-256.
"""
        ),
        code(
            r"""
checks = {}
if not results.empty:
    metrics = ["psnr_y", "ssim_y", "psnr_rgb", "ssim_rgb"]
    checks = {
        "raw_rows_360": len(results) == 360,
        "unique_tuples_360": (
            len(results.drop_duplicates(["item_id", "condition_id", "method_id"])) == 360
        ),
        "test_works_20": results.source_group_id.nunique() == 20,
        "conditions_6": set(results.condition_id) == set(config["evaluation"]["conditions"]),
        "methods_3": set(results.method_id)
        == {BICUBIC_METHOD_ID, PRETRAINED_METHOD_ID, ADAPTED_METHOD_ID},
        "metrics_finite": bool(np.isfinite(results[metrics].to_numpy()).all()),
        "aggregate_rows_18": len(aggregate) == 18,
        "paired_rows_24": len(paired) == 24,
        "qualitative_pngs_30": len(qualitative_index) == 30,
        "development_preflight": development_preflight.pages == 247
        and development_preflight.groups == 58,
        "test_preflight": test_preflight.pages == 55 and test_preflight.groups == 20,
        "checkpoints_2": len(runs) == 2
        and all(run.checkpoint_path.is_file() for run in runs.values()),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    print(checks)
    if not all(checks.values()):
        raise RuntimeError("La ejecución no supera la reconciliación completa.")
else:
    local_message = " ".join(
        [
            "Sin resultados: ejecución local de contrato completada;",
            "entrenamiento pendiente en Kaggle.",
        ]
    )
    print(local_message)
"""
        ),
        code(
            r"""
if results.empty:
    print("No hay una ejecución completa que empaquetar.")
else:
    runtime_evidence = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "seed": SEED,
        "split_sha256": split.split_sha256,
        "source_revision": split.source_revision,
        "validation": checks,
    }
    (ARTIFACT_ROOT / "runtime-evidence.json").write_text(
        json.dumps(runtime_evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    freeze = subprocess.run(
        ["uv", "pip", "freeze", "--system"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (ARTIFACT_ROOT / "environment-freeze.txt").write_text(freeze, encoding="utf-8")

    frozen_inputs = ARTIFACT_ROOT / "frozen-inputs"
    frozen_inputs.mkdir(exist_ok=True)
    for relative in (
        "configs/experiments/smb-edsr-finetuning-v1.yaml",
        "configs/degradations/staff-scale-score-v2.yaml",
        "data/adaptation/smb-edsr-finetuning-v1-split.csv",
        "data/adaptation/smb-edsr-finetuning-v1-exclusions.csv",
        "data/sources/smb.yaml",
    ):
        source = PROJECT_ROOT / relative
        shutil.copy2(source, frozen_inputs / source.name)

    manifest_path = write_run_manifest(
        ARTIFACT_ROOT,
        {
            "experiment_id": config["experiment_id"],
            "git_revision": git_revision,
            "split_sha256": split.split_sha256,
            "source_revision": split.source_revision,
            "checks": checks,
        },
    )
    archive_base = (
        Path("/kaggle/working/smb-edsr-finetuning-v1")
        if IS_KAGGLE
        else PROJECT_ROOT / "artifacts/smb-edsr-finetuning-v1"
    )
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", ARTIFACT_ROOT))
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    print(
        {
            "manifest": str(manifest_path),
            "archive": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": archive_sha256,
        }
    )
    if IS_KAGGLE:
        os.chdir("/kaggle/working")
        display(FileLink(archive_path.name))
"""
        ),
        markdown(
            """
## Conclusiones

Esta sección se completará a partir de `aggregate-metrics.csv`, `paired-bootstrap.csv` y la
revisión de los seis casos prefijados. La decisión se expresará por condición y distinguirá:

1. mejora frente al EDSR oficial;
2. mejora frente a bicúbica;
3. incertidumbre entre obras;
4. posibles alteraciones de líneas, espacios y símbolos;
5. coste temporal y límites de generalización fuera de SMB.

Hasta que la reconciliación anterior sea completamente verdadera, no hay resultado científico que
promover a la memoria.
"""
        ),
    ]
    for index, cell in enumerate(notebook.cells):
        cell.id = f"adaptation-{index:02d}"
    return notebook


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    target = root / "notebooks/04-smb-edsr-finetuning.ipynb"
    nbformat.write(build_notebook(), target)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
