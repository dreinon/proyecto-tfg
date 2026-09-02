# ruff: noqa: E501
"""Build a compact local review for the six fixed EDSR adaptation cases."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

ROLE_ORDER = (
    "reference-hr",
    "input-lr-nearest",
    "bicubic-opencv-v1",
    "edsr-baseline-official-v1",
    "edsr-smb-finetuned-v1",
)
ROLE_LABELS = {
    "reference-hr": "HR (referencia)",
    "input-lr-nearest": "LR degradada",
    "bicubic-opencv-v1": "Bicúbica",
    "edsr-baseline-official-v1": "EDSR oficial",
    "edsr-smb-finetuned-v1": "EDSR adaptado",
}
CONDITION_ORDER = (
    "x2-clean",
    "x2-moderate",
    "x2-strong",
    "x4-clean",
    "x4-moderate",
    "x4-strong",
)
DECISIONS = (
    ("improves-no-clear-defect", "Mejora sin defecto nuevo claro"),
    ("no-clear-change", "No hay cambio claro"),
    ("inherited-not-recovered", "Daño de LR no recuperado"),
    ("introduced-or-amplified", "Defecto introducido o amplificado"),
)


def image_uri(path: Path, output: Path) -> str:
    return path.relative_to(output.parent).as_posix()


def build_review(artifact_root: Path, output: Path) -> None:
    index_path = artifact_root / "evaluation/qualitative-index.csv"
    with index_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    grouped: dict[tuple[str, str, str], dict[str, Path]] = defaultdict(dict)
    for row in rows:
        key = (row["item_id"], row["source_group_id"], row["condition_id"])
        grouped[key][row["image_role"]] = artifact_root / "evaluation" / row["path"]

    cards: list[str] = []
    ordered_cases = sorted(grouped.items(), key=lambda item: CONDITION_ORDER.index(item[0][2]))
    for number, (key, images) in enumerate(ordered_cases, start=1):
        item_id, source_group_id, condition_id = key
        panes = []
        for role in ROLE_ORDER:
            path = images[role]
            panes.append(
                "<figure><figcaption>"
                + html.escape(ROLE_LABELS[role])
                + '</figcaption><a href="'
                + image_uri(path, output)
                + '" target="_blank"><img src="'
                + image_uri(path, output)
                + '" alt="'
                + html.escape(f"{condition_id}: {ROLE_LABELS[role]}")
                + '"></a></figure>'
            )
        options = "".join(
            f'<label><input type="radio" name="decision-{number}" value="{value}"> '
            f"{html.escape(label)}</label>"
            for value, label in DECISIONS
        )
        cards.append(
            f'<section class="case" data-case="{number}" data-item="{html.escape(item_id)}" '
            f'data-source="{html.escape(source_group_id)}" '
            f'data-condition="{html.escape(condition_id)}">'
            f"<h2>{number}. {html.escape(condition_id)}</h2>"
            f"<p><code>{html.escape(source_group_id)}</code> · "
            f"<code>{html.escape(item_id)}</code></p>"
            f'<div class="panes">{"".join(panes)}</div>'
            '<p class="prompt">Valora solo el EDSR adaptado. Compáralo con HR para fidelidad y '
            "con LR/EDSR oficial para atribuir el origen.</p>"
            f'<div class="choices">{options}</div>'
            f'<textarea name="notes-{number}" rows="2" '
            'placeholder="Detalle opcional: símbolos, líneas, dígitos, texto o textura"></textarea>'
            "</section>"
        )

    metadata = {
        "review_id": "smb-edsr-finetuning-v1-fixed-qualitative-review",
        "case_count": len(grouped),
        "source_index": "evaluation/qualitative-index.csv",
    }
    document = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Revisión EDSR adaptado</title><style>
body{{font:16px system-ui,sans-serif;line-height:1.45;margin:0;background:#f5f7fa;color:#17212b}}
main{{max-width:1500px;margin:auto;padding:24px}} h1{{margin-bottom:4px}} .intro{{max-width:90ch}}
.case{{background:white;border:1px solid #d6dce3;border-radius:12px;padding:18px;margin:24px 0;
box-shadow:0 2px 10px #18202a12}} .panes{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}}
figure{{margin:0}} figcaption{{font-weight:650;margin-bottom:5px}} img{{width:100%;height:260px;object-fit:contain;
background:#eee;border:1px solid #ccd3da;cursor:zoom-in}} .prompt{{font-weight:650}}
.choices{{display:flex;flex-wrap:wrap;gap:8px 18px}} .choices label{{padding:6px 9px;background:#eef3f8;border-radius:7px}}
textarea{{box-sizing:border-box;width:100%;margin-top:12px;padding:8px}} button{{font:inherit;font-weight:700;padding:10px 16px;
border:0;border-radius:8px;background:#15558d;color:white;cursor:pointer}} #status{{margin-left:12px;font-weight:650}}
@media(max-width:1000px){{.panes{{grid-template-columns:repeat(2,minmax(0,1fr))}} img{{height:320px}}}}
</style></head><body><main><h1>Revisión cualitativa: EDSR adaptado</h1>
<p class="intro">Son solo seis casos fijados antes de entrenar. No clasifiques la calidad general
de la página: decide si el <strong>EDSR adaptado</strong> mejora y si añade o amplifica un defecto.
Pulsa una imagen para verla grande. Las respuestas se guardan en este navegador.</p>
{"".join(cards)}
<button id="export">Descargar JSON</button><span id="status"></span>
<script>const meta={json.dumps(metadata)}; const key=meta.review_id;
function collect(){{return {{...meta,reviewed_at:new Date().toISOString(),assessments:[...document.querySelectorAll('.case')].map(c=>({{
item_id:c.dataset.item,source_group_id:c.dataset.source,condition_id:c.dataset.condition,
decision:c.querySelector('input:checked')?.value||'',notes:c.querySelector('textarea').value.trim()}}))}}}}
function save(){{localStorage.setItem(key,JSON.stringify(collect()));}}
document.addEventListener('change',save);document.addEventListener('input',save);
const old=JSON.parse(localStorage.getItem(key)||'null'); if(old) old.assessments.forEach((a,i)=>{{const c=document.querySelector(`[data-case="${{i+1}}"]`);
const radio=c?.querySelector(`input[value="${{a.decision}}"]`);if(radio)radio.checked=true;if(c)c.querySelector('textarea').value=a.notes||'';}});
document.querySelector('#export').onclick=()=>{{const payload=collect();const missing=payload.assessments.filter(a=>!a.decision).length;
if(missing){{document.querySelector('#status').textContent=`Faltan ${{missing}} casos.`;return;}} save();const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{{type:'application/json'}});
const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='smb-edsr-finetuning-v1-qualitative-review.json';link.click();
URL.revokeObjectURL(link.href);document.querySelector('#status').textContent='JSON descargado.';}};</script></main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/kaggle/smb-edsr-finetuning-v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/kaggle/smb-edsr-finetuning-v1-review.html"),
    )
    args = parser.parse_args()
    build_review(args.artifact_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
