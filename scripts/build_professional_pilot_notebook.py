"""Build the reader-facing external professional-pilot notebook."""

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
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook.cells = [
        markdown(
            """
# Demostrador profesional y prueba externa

## En resumen

Este cuaderno valida el flujo profesional del EDSR adaptado sobre doce obras externas que no
pertenecen a SMB. Cada HR se degrada con las seis condiciones ya congeladas y se compara con
bicúbica, EDSR oficial y EDSR adaptado. La ejecución produce 216 salidas y doce casos cualitativos
fijados sin consultar resultados.

Mientras no existan las imágenes y `source-metadata.csv`, el cuaderno solo informa del handoff
pendiente. No contiene conclusiones prefabricadas ni convierte una ejecución incompleta en
evidencia.
"""
        ),
        markdown(
            """
## Contexto y método

### Supuestos clave

- La aplicación genera derivados de consulta; no restaura una fuente histórica.
- Las doce obras son independientes y se usan exclusivamente como test externo.
- La LR se genera sintéticamente desde HR con el protocolo normalizado por pentagrama ya fijado.
- El resultado permite estudiar transferencia entre corpus bajo degradación emparejada, no LR
  real adquirida de forma natural ni generalización universal.
- No se cambian pesos, degradaciones, umbrales ni selección después de abrir las salidas.
"""
        ),
        markdown(
            """
## Preparación

Coloca las páginas en `data/raw/professional-pilot-v1/` y copia allí la plantilla
`docs/templates/professional-pilot-source-metadata.csv` con el nombre `source-metadata.csv`.
Completa los identificadores, la procedencia institucional y los derechos antes de ejecutar. Los
ficheros de entrada permanecen ignorados por Git. El corpus SJMA actual es privado. Su
representante legal ha autorizado el procesamiento en un entorno privado de Kaggle, pero la ruta
local sigue siendo la predeterminada y ninguna entrada o salida debe hacerse pública.
"""
        ),
        code(
            """
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from IPython.display import FileLink, display

from score_super_resolution.external_pilot import (
    ARTIFACT_ROOT_RELATIVE_PATH,
    SOURCE_ROOT_RELATIVE_PATH,
    ExternalPilotError,
    analyze_external_pilot,
    evaluate_external_pilot,
    freeze_external_pilot_manifest,
    load_external_pilot_pages,
)


def discover_project_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    raise RuntimeError("No se encuentra la raíz del proyecto.")


PROJECT_ROOT = discover_project_root()
SOURCE_ROOT = PROJECT_ROOT / SOURCE_ROOT_RELATIVE_PATH
ARTIFACT_ROOT = PROJECT_ROOT / ARTIFACT_ROOT_RELATIVE_PATH
RUN_EXTERNAL_PILOT = os.environ.get("RUN_EXTERNAL_PILOT") == "1"
{
    "project_root": str(PROJECT_ROOT),
    "source_root": str(SOURCE_ROOT),
    "run_external_pilot": RUN_EXTERNAL_PILOT,
}
"""
        ),
        markdown("## Datos"),
        code(
            """
pages = None
input_error = None
try:
    pages = load_external_pilot_pages(PROJECT_ROOT, source_root=SOURCE_ROOT)
    manifest_path, manifest_sha256 = freeze_external_pilot_manifest(pages, ARTIFACT_ROOT)
except ExternalPilotError as error:
    input_error = str(error)

if pages is None:
    print(f"ENTRADA PENDIENTE: {input_error}")
else:
    inventory = pd.DataFrame(page.evidence_record() for page in pages)
    display(
        inventory[
            [
                "role",
                "work_id",
                "genre",
                "instrument",
                "orientation",
                "source_type",
                "notation_density",
                "document_condition",
                "width",
                "height",
                "staff_spacing_px",
            ]
        ]
    )
    print(
        {
            "works": len(inventory),
            "test_works": int((inventory.role == "test").sum()),
            "manifest_sha256": manifest_sha256,
            "manifest_path": str(manifest_path),
        }
    )
"""
        ),
        markdown(
            """
## Resultados

La celda siguiente solo abre los resultados cuando `RUN_EXTERNAL_PILOT=1`. En VS Code puede
definirse la variable antes de iniciar el kernel o sustituirse temporalmente el valor en la celda
de preparación. La ejecución es reanudable a partir de `raw-metrics.csv`.
"""
        ),
        code(
            """
results = None
if pages is None:
    print("La evaluación no puede comenzar hasta cerrar el manifiesto de entrada.")
elif not RUN_EXTERNAL_PILOT:
    print("Preflight correcto. Define RUN_EXTERNAL_PILOT=1 para ejecutar las 216 salidas.")
else:
    results = evaluate_external_pilot(
        PROJECT_ROOT,
        pages,
        output_root=ARTIFACT_ROOT,
    )
    print(
        {
            "rows": len(results),
            "works": results.source_group_id.nunique(),
            "conditions": results.condition_id.nunique(),
            "methods": results.method_id.nunique(),
        }
    )
"""
        ),
        code(
            """
aggregate = None
paired = None
if results is not None:
    aggregate, paired = analyze_external_pilot(results, ARTIFACT_ROOT)
    display(aggregate)
    display(paired[paired.comparator_id == "edsr-baseline-official-v1"])
else:
    print("Sin resultados ejecutados: no se calculan ni muestran conclusiones.")
"""
        ),
        markdown("## Revisión cualitativa"),
        code(
            """
review_path = ARTIFACT_ROOT / "professional-pilot-v1-review.html"
if results is not None:
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/build_professional_pilot_review.py"),
            "--artifact-root",
            str(ARTIFACT_ROOT),
            "--output",
            str(review_path),
        ],
        check=True,
    )
    display(FileLink(review_path))
    print("Completa los doce casos y descarga el JSON; no se infiere ninguna respuesta.")
else:
    print("La revisión cualitativa se genera después de reconciliar las 216 salidas.")
"""
        ),
        markdown("## Comprobaciones"),
        code(
            """
if results is not None:
    checks = {
        "exactly_216_rows": len(results) == 216,
        "twelve_independent_works": results.source_group_id.nunique() == 12,
        "six_conditions": results.condition_id.nunique() == 6,
        "three_methods": results.method_id.nunique() == 3,
        "no_duplicate_tuple": not results.duplicated(
            ["source_group_id", "condition_id", "method_id"]
        ).any(),
        "metrics_finite": results[["psnr_y", "ssim_y", "psnr_rgb", "ssim_rgb", "runtime_seconds"]]
        .notna()
        .all()
        .all(),
        "twelve_qualitative_cases": len(list((ARTIFACT_ROOT / "qualitative").glob("*/*"))) == 12,
        "evaluation_identity_present": (ARTIFACT_ROOT / "evaluation-identity.json").is_file(),
        "runtime_evidence_present": (ARTIFACT_ROOT / "runtime-evidence.json").is_file(),
        "artifact_manifest_present": (ARTIFACT_ROOT / "artifact-manifest.json").is_file(),
        "review_interface_present": review_path.is_file(),
    }
    print(checks)
    if not all(checks.values()):
        raise RuntimeError("La prueba externa no ha reconciliado todos sus controles.")
else:
    print("Comprobaciones de resultados pendientes de ejecución.")
"""
        ),
        markdown("## Paquete descargable"),
        code(
            """
if results is not None:
    archive_path = Path(
        shutil.make_archive(
            str(ARTIFACT_ROOT),
            "zip",
            root_dir=ARTIFACT_ROOT.parent,
            base_dir=ARTIFACT_ROOT.name,
        )
    )
    display(FileLink(archive_path))
    print({"archive": str(archive_path), "bytes": archive_path.stat().st_size})
else:
    print("El paquete se genera solo después de completar y comprobar la ejecución.")
"""
        ),
        markdown(
            """
## Conclusiones

Las conclusiones se redactarán únicamente después de reconciliar las 216 salidas, revisar las
tablas emparejadas y completar los doce casos cualitativos. Una mejora métrica no implica que el
derivado sea correcto para edición, conservación u OMR.
"""
        ),
    ]
    return notebook


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "notebooks/05-professional-demonstrator-validation.ipynb"
    nbformat.write(build_notebook(), output)
    print(output)


if __name__ == "__main__":
    main()
