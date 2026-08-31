# ruff: noqa: E501, RUF001
"""Build the canonical portable technical report for the final SMB v2 analysis."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

METHOD_LABELS = {
    "bicubic-opencv-v1": "Bicúbica",
    "edsr-baseline-official-v1": "EDSR",
    "swinir-lightweight-official-v1": "SwinIR",
}
METRIC_LABELS = {"psnr_y": "PSNR-Y", "ssim_y": "SSIM-Y"}


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    metric_definitions: list[str],
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM read_csv_auto('{path}', header = true)",
            "description": description,
            "filters": [
                "Frozen SMB v2 analysis outputs only",
                "No post-outcome tuning or exclusions",
            ],
            "metric_definitions": metric_definitions,
        },
    }


def build_artifact(analysis_root: Path) -> dict[str, Any]:
    aggregate = pd.read_csv(analysis_root / "aggregate-metrics.csv")
    paired = pd.read_csv(analysis_root / "paired-bootstrap.csv")
    runtime = pd.read_csv(analysis_root / "runtime-summary.csv")
    qualitative_status = pd.read_csv(analysis_root / "qualitative-status.csv")
    qualitative_flags = pd.read_csv(analysis_root / "qualitative-flags.csv")
    summary = json.loads((analysis_root / "integrated-summary.json").read_text(encoding="utf-8"))

    aggregate["method"] = aggregate["method_id"].map(METHOD_LABELS)
    aggregate["condition"] = aggregate["condition_id"]
    paired["method"] = paired["method_id"].map(METHOD_LABELS)
    paired["comparator"] = paired["comparator_id"].map(METHOD_LABELS)
    paired["metric_label"] = paired["metric"].map(METRIC_LABELS)
    paired["ci95"] = paired.apply(lambda row: f"[{row.ci95_low:.4f}, {row.ci95_high:.4f}]", axis=1)
    runtime["method"] = runtime["method_id"].map(METHOD_LABELS)
    runtime["condition"] = runtime["condition_id"]
    qualitative_status["method"] = qualitative_status["method_id"].map(METHOD_LABELS)
    qualitative_status["status_label"] = qualitative_status["status"].map(
        {"no-clear-issue": "Sin defecto claro", "issues-observed": "Defectos observados"}
    )
    qualitative_flags["method"] = qualitative_flags["method_id"].map(METHOD_LABELS)

    baseline_pairs = paired[paired["comparator_id"] == "bicubic-opencv-v1"].copy()
    direct_pairs = paired[paired["comparator_id"] == "edsr-baseline-official-v1"].copy()
    generated_at = datetime.now(UTC).isoformat()
    metrics_source = _source(
        "smb-v2-metrics",
        "Agregados finales SMB v2",
        "artifacts/kaggle/phase3-smb-analysis-v2/final/aggregate-metrics.csv",
        "Carga los agregados por condición y método generados desde las 1.152 tuplas reconciliadas.",
        [
            "PSNR-Y: PSNR sobre luminancia BT.601; media de 64 obras dentro de cada condición y método.",
            "SSIM-Y: SSIM Wang et al. sobre luminancia BT.601; media de 64 obras dentro de cada condición y método.",
            "Diferencia pareada: métrica del método menos métrica del comparador para la misma obra y condición.",
            "IC 95%: percentiles 2,5 y 97,5 de 2.000 remuestreos con reemplazo de las 64 obras independientes.",
        ],
    )
    paired_source = _source(
        "smb-v2-paired",
        "Bootstrap pareado SMB v2",
        "artifacts/kaggle/phase3-smb-analysis-v2/final/paired-bootstrap.csv",
        "Carga las diferencias pareadas y sus intervalos por obra independiente.",
        [
            "Diferencia pareada: métrica del método menos métrica del comparador para la misma obra y condición.",
            "IC 95%: percentiles 2,5 y 97,5 de 2.000 remuestreos con reemplazo de las 64 obras independientes.",
        ],
    )
    runtime_source = _source(
        "smb-v2-runtime",
        "Tiempos de inferencia SMB v2",
        "artifacts/kaggle/phase3-smb-analysis-v2/final/runtime-summary.csv",
        "Resume la mediana del tiempo medido por método y condición en la ejecución Kaggle con dos Tesla T4.",
        [
            "Runtime: mediana de segundos por página y método dentro de cada condición.",
            "Cocientes de tiempo: mediana del método dividida por la mediana del comparador en la misma condición.",
        ],
    )
    qualitative_source = _source(
        "smb-v2-qualitative",
        "Revisión humana cualitativa SMB v2",
        "artifacts/kaggle/phase3-smb-analysis-v2/final/qualitative-status.csv",
        "Resume 18 valoraciones humanas sobre seis casos fijados antes de observar los resultados.",
        [
            "Estado cualitativo: sin defecto claro o defectos observados frente a la referencia HR.",
            "Los recuentos describen los seis casos revisados por método y no estiman prevalencia en SMB.",
        ],
    )
    qualitative_flags_source = _source(
        "smb-v2-qualitative-flags",
        "Etiquetas cualitativas SMB v2",
        "artifacts/kaggle/phase3-smb-analysis-v2/final/qualitative-flags.csv",
        "Carga los tipos de defecto marcados en los seis casos fijados por método.",
        [
            "Un mismo caso puede tener varias etiquetas y los recuentos no estiman prevalencia.",
        ],
    )
    sources = [
        metrics_source,
        paired_source,
        runtime_source,
        qualitative_source,
        qualitative_flags_source,
    ]

    title = "Evaluación final SMB v2 de superresolución musical"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "Síntesis técnica reproducible de fidelidad, incertidumbre, coste y fallos de notación.",
        "generatedAt": generated_at,
        "filters": [],
        "cards": [],
        "charts": [
            {
                "id": "psnr-chart",
                "title": "PSNR-Y medio por condición y método",
                "subtitle": "64 obras independientes por barra; valores mayores indican mayor fidelidad de luminancia.",
                "showDescription": True,
                "headerMarkdown": "Las ventajas grandes de los modelos aprendidos se concentran en x2 limpio y moderado; en x2 fuerte la diferencia frente a bicúbica es pequeña.",
                "intent": "comparison",
                "question": "¿Cómo cambia el PSNR-Y medio entre métodos dentro de cada condición?",
                "rationale": "Las barras agrupadas comparan tres métodos en seis categorías discretas con la misma unidad.",
                "comparisonContext": {
                    "baseline": "Bicúbica",
                    "denominator": "64 obras independientes por condición",
                    "grain": "condición-método",
                    "unit": "dB",
                },
                "type": "bar",
                "dataset": "aggregate",
                "sourceId": "smb-v2-metrics",
                "encodings": {
                    "x": {"field": "condition", "type": "ordinal", "label": "Condición"},
                    "y": {
                        "field": "psnr_y_mean",
                        "type": "quantitative",
                        "label": "PSNR-Y medio",
                        "unit": "dB",
                    },
                    "color": {"field": "method", "type": "nominal", "label": "Método"},
                    "tooltip": [
                        {"field": "sources", "type": "quantitative", "label": "Obras"},
                        {"field": "profile", "type": "nominal", "label": "Severidad"},
                    ],
                },
                "combinationRationale": "El color distingue métodos y no duplica la condición del eje X.",
                "valueFormat": "number",
                "unit": "dB",
                "layout": "full",
            },
            {
                "id": "ssim-chart",
                "title": "SSIM-Y medio por condición y método",
                "subtitle": "64 obras independientes por barra; escala [0, 1], donde valores mayores indican mayor similitud estructural.",
                "showDescription": True,
                "headerMarkdown": "En x4 EDSR y SwinIR superan ampliamente a bicúbica, pero la inspección humana impide interpretar esa ventaja como preservación musical garantizada.",
                "intent": "comparison",
                "question": "¿Cómo cambia el SSIM-Y medio entre métodos dentro de cada condición?",
                "rationale": "Las barras agrupadas permiten comparar el mismo índice acotado para tres métodos y seis condiciones.",
                "comparisonContext": {
                    "baseline": "Bicúbica",
                    "denominator": "64 obras independientes por condición",
                    "grain": "condición-método",
                    "unit": "índice [0, 1]",
                },
                "type": "bar",
                "dataset": "aggregate",
                "sourceId": "smb-v2-metrics",
                "encodings": {
                    "x": {"field": "condition", "type": "ordinal", "label": "Condición"},
                    "y": {
                        "field": "ssim_y_mean",
                        "type": "quantitative",
                        "label": "SSIM-Y medio",
                    },
                    "color": {"field": "method", "type": "nominal", "label": "Método"},
                    "tooltip": [
                        {"field": "sources", "type": "quantitative", "label": "Obras"},
                        {"field": "profile", "type": "nominal", "label": "Severidad"},
                    ],
                },
                "combinationRationale": "El color distingue métodos y no duplica la condición del eje X.",
                "valueFormat": "number",
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "baseline-table",
                "title": "Diferencias pareadas frente a bicúbica",
                "subtitle": "Método menos bicúbica; IC percentil del 95% con 2.000 remuestreos de 64 obras.",
                "dataset": "paired_baseline",
                "sourceId": "smb-v2-paired",
                "density": "comfortable",
                "layout": "full",
                "defaultSort": {"field": "condition_id", "direction": "asc"},
                "columns": [
                    {"field": "condition_id", "label": "Condición", "type": "text"},
                    {"field": "metric_label", "label": "Métrica", "type": "text"},
                    {"field": "method", "label": "Método", "type": "text"},
                    {
                        "field": "mean_delta",
                        "label": "Diferencia media",
                        "format": "number",
                        "movement": True,
                    },
                    {"field": "ci95", "label": "IC 95%", "type": "text"},
                ],
            },
            {
                "id": "direct-table",
                "title": "Comparación pareada SwinIR frente a EDSR",
                "subtitle": "Valores positivos favorecen a SwinIR; valores negativos favorecen a EDSR.",
                "dataset": "paired_direct",
                "sourceId": "smb-v2-paired",
                "density": "comfortable",
                "layout": "full",
                "defaultSort": {"field": "condition_id", "direction": "asc"},
                "columns": [
                    {"field": "condition_id", "label": "Condición", "type": "text"},
                    {"field": "metric_label", "label": "Métrica", "type": "text"},
                    {
                        "field": "mean_delta",
                        "label": "SwinIR − EDSR",
                        "format": "number",
                        "movement": True,
                    },
                    {"field": "ci95", "label": "IC 95%", "type": "text"},
                ],
            },
            {
                "id": "runtime-table",
                "title": "Tiempo de inferencia por condición",
                "subtitle": "Mediana por página en la ejecución final con dos Tesla T4; no es una medida portable a otro hardware.",
                "dataset": "runtime",
                "sourceId": "smb-v2-runtime",
                "density": "comfortable",
                "layout": "full",
                "defaultSort": {"field": "condition_id", "direction": "asc"},
                "columns": [
                    {"field": "condition_id", "label": "Condición", "type": "text"},
                    {"field": "method", "label": "Método", "type": "text"},
                    {
                        "field": "runtime_seconds_median",
                        "label": "Mediana",
                        "format": "number",
                        "unit": "s/página",
                    },
                    {
                        "field": "runtime_vs_edsr",
                        "label": "× EDSR",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "qualitative-table",
                "title": "Dictámenes de los seis casos cualitativos fijados",
                "subtitle": "Recuentos descriptivos por método; no son tasas de prevalencia en SMB.",
                "dataset": "qualitative_status",
                "sourceId": "smb-v2-qualitative",
                "density": "spacious",
                "layout": "full",
                "defaultSort": {"field": "method", "direction": "asc"},
                "columns": [
                    {"field": "method", "label": "Método", "type": "text"},
                    {"field": "status_label", "label": "Dictamen", "type": "text"},
                    {
                        "field": "fixed_case_count",
                        "label": "Casos fijados",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "flags-table",
                "title": "Tipos de defecto observados en los casos fijados",
                "subtitle": "Un mismo caso puede tener varias etiquetas; los recuentos no suman seis.",
                "dataset": "qualitative_flags",
                "sourceId": "smb-v2-qualitative-flags",
                "density": "comfortable",
                "layout": "full",
                "defaultSort": {"field": "fixed_case_count", "direction": "desc"},
                "columns": [
                    {"field": "method", "label": "Método", "type": "text"},
                    {"field": "flag_id", "label": "Tipo de defecto", "type": "text"},
                    {
                        "field": "fixed_case_count",
                        "label": "Casos etiquetados",
                        "format": "number",
                    },
                ],
            },
        ],
        "sources": sources,
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {
                "id": "technical-summary",
                "type": "markdown",
                "body": (
                    "## La mejora de fidelidad es real, pero no equivale a restauración musical segura\n\n"
                    "Los dos modelos aprendidos superan a la interpolación bicúbica en PSNR-Y y SSIM-Y "
                    "en las seis condiciones: los 24 intervalos pareados frente a bicúbica quedan por encima "
                    "de cero. La magnitud, sin embargo, depende mucho del régimen. En x2 limpio y moderado "
                    "las mejoras son amplias; en x2 fuerte se reducen a aproximadamente +0,20 dB para EDSR y "
                    "+0,18 dB para SwinIR.\n\n"
                    "SwinIR lidera claramente x2 limpio y moderado, pero EDSR iguala o supera su fidelidad en "
                    "los regímenes fuertes y en x4 moderado/fuerte, con un coste entre 5,2 y 8,5 veces menor "
                    "que SwinIR en esta ejecución. La revisión humana detectó defectos en cuatro de seis casos "
                    "fijados para cada modelo aprendido y en dos de seis para bicúbica. Esta muestra cualitativa "
                    "explica fallos, pero no estima su prevalencia.\n\n"
                    "**Conclusión profesional:** EDSR ofrece el compromiso más defendible entre fidelidad y "
                    "coste para mejora visual asistida, siempre conservando el original y con validación humana. "
                    "Ningún método evaluado debe presentarse como restauración semántica automática ni como "
                    "entrada fiable para OMR sin una evaluación específica adicional."
                ),
            },
            {
                "id": "fidelity-section",
                "type": "markdown",
                "body": (
                    "## Los modelos aprendidos dominan en fidelidad, sobre todo fuera de x2 fuerte\n\n"
                    "Las dos figuras comparan métodos únicamente dentro de la misma escala y severidad. El salto "
                    "frente a bicúbica es máximo en x2 limpio/moderado y sigue siendo notable en x4. La aparente "
                    "ventaja cuantitativa no elimina la necesidad de inspección musical: PSNR y SSIM premian la "
                    "similitud de píxel y estructura, no garantizan que alteraciones pequeñas con significado "
                    "musical hayan desaparecido."
                ),
                "sourceId": "smb-v2-metrics",
            },
            {"id": "psnr-block", "type": "chart", "chartId": "psnr-chart", "layout": "full"},
            {"id": "ssim-block", "type": "chart", "chartId": "ssim-chart", "layout": "full"},
            {
                "id": "uncertainty-section",
                "type": "markdown",
                "body": (
                    "## El remuestreo confirma la dirección, pero también revela diferencias prácticas pequeñas\n\n"
                    "Los intervalos se calculan sobre obras independientes y conservan el emparejamiento entre "
                    "métodos. Todos los contrastes de los modelos aprendidos frente a bicúbica son positivos en "
                    "ambas métricas, aunque x2 fuerte presenta ganancias de PSNR pequeñas. Entre modelos, SwinIR "
                    "es superior en x2 limpio/moderado y x4 limpio; EDSR es superior en SSIM-Y bajo x2 fuerte y "
                    "x4 moderado/fuerte, y también en PSNR-Y x4 fuerte. Los intervalos que cruzan cero se conservan "
                    "como resultados no concluyentes, no como empates demostrados."
                ),
                "sourceId": "smb-v2-paired",
            },
            {"id": "baseline-block", "type": "table", "tableId": "baseline-table"},
            {"id": "direct-block", "type": "table", "tableId": "direct-table"},
            {
                "id": "scope-section",
                "type": "markdown",
                "body": (
                    "## Alcance, datos y definiciones\n\n"
                    "El análisis utiliza 64 páginas de 64 obras distintas del único split oficial `test` de SMB, "
                    "seleccionadas antes de observar los resultados y disjuntas de la ejecución v1. Cada obra se "
                    "evalúa en seis condiciones: x2/x4 combinados con degradación limpia, moderada y fuerte, "
                    "normalizada por el espaciado de pentagrama estimado solo desde la entrada. Bicúbica, EDSR y "
                    "SwinIR procesan exactamente los mismos pares LR/HR. PSNR-Y y SSIM-Y son las métricas "
                    "primarias; RGB y tiempos se conservan como diagnósticos. La v1 queda como evidencia de "
                    "desarrollo y corrección del protocolo, no como confirmación final."
                ),
            },
            {
                "id": "method-section",
                "type": "markdown",
                "body": (
                    "## Diseño inferencial y robustez\n\n"
                    "Para cada condición se calcula primero la diferencia entre métodos sobre la misma obra. Se "
                    "remuestrean con reemplazo las 64 obras 2.000 veces con semilla 20260831 y se informa del "
                    "intervalo percentil del 95 % de la diferencia media. El diseño evita pseudorreplicación, "
                    "mantiene el denominador pareado y no introduce decisiones posteriores a los resultados. "
                    "Se verificaron las 1.152 tuplas esperadas, 384 trazas de degradación, los hashes de salida, "
                    "la identidad del preflight y los 30 PNG cualitativos."
                ),
            },
            {
                "id": "runtime-section",
                "type": "markdown",
                "body": (
                    "## EDSR conserva la mayor parte de la utilidad con mucho menos coste que SwinIR\n\n"
                    "En las dos Tesla T4, EDSR tarda aproximadamente 0,36–0,67 s por página y SwinIR "
                    "1,90–5,67 s. SwinIR resulta entre 5,2 y 8,5 veces más lento que EDSR; bicúbica permanece "
                    "dos órdenes de magnitud por debajo. Los tiempos sirven para comparar esta ejecución y no "
                    "deben extrapolarse directamente a otra GPU, CPU, resolución o implementación."
                ),
                "sourceId": "smb-v2-runtime",
            },
            {"id": "runtime-block", "type": "table", "tableId": "runtime-table"},
            {
                "id": "qualitative-section",
                "type": "markdown",
                "body": (
                    "## La inspección musical impide convertir la ventaja métrica en un ganador universal\n\n"
                    "La revisión fijó un caso por condición y evaluó cada salida frente a HR, usando LR como "
                    "contexto causal. Bicúbica no mostró defectos claros en cuatro casos; EDSR y SwinIR, en dos "
                    "cada uno. En los modelos aprendidos se observaron simplificación o pérdida de símbolos, "
                    "corrupción de texto/dígitos y textura o curvatura artificial. En x4 fuerte, parte del daño "
                    "bicúbico se atribuyó a la degradación no recuperada, mientras que las notas de EDSR/SwinIR "
                    "describieron reconstrucción añadida. Estos resultados son ejemplos explicativos, no tasas "
                    "de error del conjunto completo."
                ),
            },
            {"id": "qualitative-block", "type": "table", "tableId": "qualitative-table"},
            {"id": "flags-block", "type": "table", "tableId": "flags-table"},
            {
                "id": "limitations-section",
                "type": "markdown",
                "body": (
                    "## Límites que condicionan cualquier uso profesional\n\n"
                    "- La muestra final cubre solo páginas SMB con escala de pentagrama medible y degradaciones "
                    "sintéticas controladas; no valida escaneos reales, fotografías, PDF ni otros archivos.\n"
                    "- La revisión cualitativa incluye seis casos, un único revisor no cegado y métodos "
                    "identificados; detecta mecanismos de fallo, pero no permite estimar frecuencia ni acuerdo.\n"
                    "- PSNR/SSIM no miden directamente legibilidad, corrección musical u OMR.\n"
                    "- Los modelos se usan preentrenados en imagen natural y no se adaptan a la geometría de "
                    "pentagramas; esto explica plausiblemente las texturas curvas observadas, pero el experimento "
                    "no identifica causalmente su origen interno.\n"
                    "- Los tiempos pertenecen a una ejecución concreta con dos Tesla T4 y no representan un SLA."
                ),
            },
            {
                "id": "next-section",
                "type": "markdown",
                "body": (
                    "## Recomendación y próximos pasos\n\n"
                    "1. Usar EDSR como candidato principal de mejora visual asistida cuando el coste importe, "
                    "manteniendo bicúbica como referencia y alternativa segura.\n"
                    "2. Conservar y mostrar siempre la imagen original; no sustituir el documento maestro por la "
                    "salida reconstruida.\n"
                    "3. Exigir inspección humana en símbolos pequeños, alteraciones, texto, dígitos y líneas de "
                    "pentagrama antes de publicación o preservación.\n"
                    "4. Declarar NO-GO para ajuste fino, OMR y pruebas de escaneo real antes del depósito; solo "
                    "reabrirlos como trabajo futuro si el núcleo y la memoria quedan cerrados.\n"
                    "5. Convertir estas tablas, figuras y límites en la sección de resultados y discusión de la "
                    "memoria, manteniendo la separación entre v1 de desarrollo y v2 final."
                ),
            },
            {
                "id": "questions-section",
                "type": "markdown",
                "body": (
                    "## Preguntas que quedan abiertas\n\n"
                    "La principal incertidumbre externa es si el mismo compromiso se mantiene en degradaciones "
                    "reales y en tareas posteriores como OMR. También queda por medir si un modelo entrenado con "
                    "restricciones específicas de pentagrama reduce artefactos sin sacrificar coste. Ambas "
                    "preguntas son futuras extensiones y no condicionan la conclusión del núcleo controlado."
                ),
            },
        ],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "aggregate": _records(aggregate),
                "paired_baseline": _records(baseline_pairs),
                "paired_direct": _records(direct_pairs),
                "runtime": _records(runtime),
                "qualitative_status": _records(qualitative_status),
                "qualitative_flags": _records(qualitative_flags),
            },
            "accessIssues": [],
        },
        "sources": sources,
        "package_info": {
            "originUrl": "artifact://smb-v2-final-analysis",
            "controls": {"edit": False, "refresh": False},
        },
        "analysis_context": {
            "summary": summary,
            "report_structure": [
                "technical summary",
                "key findings with visual evidence",
                "scope, data, and metric definitions",
                "methodology and validation",
                "limitations and uncertainty",
                "recommended next steps",
                "further questions",
            ],
            "chart_map": [
                {
                    "section": "fidelity",
                    "question": "PSNR-Y comparison within condition",
                    "family": "comparison",
                    "type": "bar",
                    "fields": ["condition", "psnr_y_mean", "method"],
                    "palette_policy": "relaxed multi-category, three methods",
                },
                {
                    "section": "fidelity",
                    "question": "SSIM-Y comparison within condition",
                    "family": "comparison",
                    "type": "bar",
                    "fields": ["condition", "ssim_y_mean", "method"],
                    "palette_policy": "relaxed multi-category, three methods",
                },
            ],
            "omissions": [
                "Bootstrap intervals use exact tables because the canonical chart contract has no interval-mark chart.",
                "Runtime uses a table because the methods span three orders of magnitude and exact trade-offs matter.",
                "Qualitative counts use tables to avoid implying population prevalence from six fixed cases.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("artifacts/kaggle/phase3-smb-analysis-v2/final"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/kaggle/phase3-smb-analysis-v2/final/report-artifact.json"),
    )
    args = parser.parse_args()
    artifact = build_artifact(args.analysis_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
