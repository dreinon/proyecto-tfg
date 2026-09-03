# ruff: noqa: E501
"""Build the fixed local review for the twelve external professional-pilot cases."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from score_super_resolution.edsr_finetuning import write_run_manifest

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
ACCEPTANCE = (
    ("acceptable", "Aceptable para consulta"),
    ("acceptable-with-reservations", "Aceptable con reservas"),
    ("rejected", "Rechazado"),
)
ATTRIBUTION = (
    ("no-clear-defect", "Sin defecto nuevo claro"),
    ("inherited-not-recovered", "Daño de LR no recuperado"),
    ("introduced-or-amplified", "Defecto introducido o amplificado"),
    ("uncertain", "Origen incierto"),
)
REVIEW_ID = "professional-pilot-v1-fixed-qualitative-review"


def _image_uri(path: Path, output: Path) -> str:
    return path.relative_to(output.parent).as_posix()


def build_review(artifact_root: Path, output: Path) -> None:
    index_path = artifact_root / "qualitative-index.csv"
    manifest_path = artifact_root / "input-manifest.csv"
    identity_path = artifact_root / "evaluation-identity.json"
    with index_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    with manifest_path.open(encoding="utf-8", newline="") as source:
        source_rows = {row["work_id"]: row for row in csv.DictReader(source)}
    identity = json.loads(identity_path.read_text(encoding="utf-8"))

    grouped: dict[tuple[str, str, str], dict[str, Path]] = defaultdict(dict)
    for row in rows:
        key = (row["item_id"], row["source_group_id"], row["condition_id"])
        grouped[key][row["image_role"]] = artifact_root / row["path"]
    if len(grouped) != 12 or any(set(images) != set(ROLE_ORDER) for images in grouped.values()):
        raise ValueError("the external review requires twelve complete fixed cases")
    counts = Counter(key[2] for key in grouped)
    if any(counts[condition] != 2 for condition in CONDITION_ORDER):
        raise ValueError("the external review requires two cases per condition")

    cards: list[str] = []
    ordered_cases = sorted(
        grouped.items(), key=lambda item: (CONDITION_ORDER.index(item[0][2]), item[0][1])
    )
    for number, (key, images) in enumerate(ordered_cases, start=1):
        item_id, source_group_id, condition_id = key
        source = source_rows[source_group_id]
        panes = []
        for role in ROLE_ORDER:
            uri = _image_uri(images[role], output)
            label = ROLE_LABELS[role]
            panes.append(
                f'<figure><figcaption>{html.escape(label)}</figcaption><a href="{uri}" '
                f'target="_blank"><img src="{uri}" alt="{html.escape(condition_id + ": " + label)}">'
                "</a></figure>"
            )
        acceptance = "".join(
            f'<label><input type="radio" name="acceptance-{number}" value="{value}"> '
            f"{html.escape(label)}</label>"
            for value, label in ACCEPTANCE
        )
        attribution = "".join(
            f'<label><input type="radio" name="attribution-{number}" value="{value}"> '
            f"{html.escape(label)}</label>"
            for value, label in ATTRIBUTION
        )
        attributes = " · ".join(
            html.escape(source[field])
            for field in (
                "genre",
                "instrument",
                "orientation",
                "source_type",
                "notation_density",
                "document_condition",
            )
        )
        cards.append(
            f'<section class="case" data-case="{number}" data-item="{html.escape(item_id)}" '
            f'data-source="{html.escape(source_group_id)}" data-condition="{html.escape(condition_id)}">'
            f"<h2>{number}. {html.escape(condition_id)}</h2><p><code>{html.escape(source_group_id)}</code> · {attributes}</p>"
            f'<div class="panes">{"".join(panes)}</div>'
            '<p class="prompt">¿El EDSR adaptado sirve como derivado de consulta en este caso?</p>'
            f'<div class="choices">{acceptance}</div>'
            '<p class="prompt">Si existe daño, atribúyelo comparando HR, LR y EDSR oficial.</p>'
            f'<div class="choices">{attribution}</div>'
            f'<textarea name="notes-{number}" rows="3" placeholder="Describe líneas, símbolos pequeños, uniones, texto, dígitos, halos o marcas inventadas"></textarea>'
            "</section>"
        )

    metadata = {
        "review_id": REVIEW_ID,
        "case_count": len(grouped),
        "input_manifest_sha256": identity["input_manifest_sha256"],
        "source_index": "qualitative-index.csv",
    }
    document = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Prueba profesional externa</title><style>
body{{font:16px system-ui,sans-serif;line-height:1.45;margin:0;background:#f0ece2;color:#1d1b18}}
main{{max-width:1540px;margin:auto;padding:24px}} h1{{font-family:Georgia,serif;margin-bottom:4px}} .intro{{max-width:95ch}}
.case{{background:#fffdf7;border:1px solid #c9beaa;border-radius:10px;padding:18px;margin:24px 0;box-shadow:0 2px 12px #3b2b1714}}
.panes{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}} figure{{margin:0}} figcaption{{font-weight:700;margin-bottom:5px}}
img{{width:100%;height:280px;object-fit:contain;background:#e7e1d6;border:1px solid #c9beaa;cursor:zoom-in}}
.prompt{{font-weight:700;margin-bottom:7px}} .choices{{display:flex;flex-wrap:wrap;gap:8px 16px}}
.choices label{{padding:7px 10px;background:#eee7da;border-radius:6px}} textarea{{box-sizing:border-box;width:100%;margin-top:14px;padding:9px}}
button{{font:inherit;font-weight:750;padding:11px 17px;border:0;border-radius:7px;background:#923f26;color:white;cursor:pointer}}
#status{{margin-left:12px;font-weight:700}} @media(max-width:1000px){{.panes{{grid-template-columns:repeat(2,minmax(0,1fr))}}img{{height:330px}}}}
</style></head><body><main><h1>Prueba profesional externa</h1>
<p class="intro">Revisa los doce casos fijados antes de ver las salidas. La HR indica fidelidad; la LR muestra qué información se perdió; EDSR oficial permite atribuir el efecto de la adaptación. La aceptación se limita a un <strong>derivado de consulta</strong>, no a restauración, edición ni interpretación automática. Pulsa cualquier imagen para ampliarla.</p>
{"".join(cards)}
<button id="export">Descargar JSON</button><span id="status"></span>
<script>const meta={json.dumps(metadata)};const key=meta.review_id;
function collect(){{return {{...meta,reviewed_at:new Date().toISOString(),assessments:[...document.querySelectorAll('.case')].map(c=>({{
item_id:c.dataset.item,source_group_id:c.dataset.source,condition_id:c.dataset.condition,
acceptance:c.querySelector(`input[name="acceptance-${{c.dataset.case}}"]:checked`)?.value||'',
attribution:c.querySelector(`input[name="attribution-${{c.dataset.case}}"]:checked`)?.value||'',notes:c.querySelector('textarea').value.trim()}}))}}}}
function save(){{localStorage.setItem(key,JSON.stringify(collect()));}}document.addEventListener('change',save);document.addEventListener('input',save);
const old=JSON.parse(localStorage.getItem(key)||'null');if(old)old.assessments.forEach((a,i)=>{{const c=document.querySelector(`[data-case="${{i+1}}"]`);for(const field of ['acceptance','attribution']){{const radio=c?.querySelector(`input[name="${{field}}-${{i+1}}"]`+`[value="${{a[field]}}"]`);if(radio)radio.checked=true;}}if(c)c.querySelector('textarea').value=a.notes||'';}});
document.querySelector('#export').onclick=()=>{{const payload=collect();const missing=payload.assessments.filter(a=>!a.acceptance||!a.attribution).length;
if(missing){{document.querySelector('#status').textContent=`Faltan ${{missing}} casos.`;return;}}save();const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{{type:'application/json'}});
const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='professional-pilot-v1-qualitative-review.json';link.click();URL.revokeObjectURL(link.href);document.querySelector('#status').textContent='JSON descargado.';}};</script>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    manifest_path = artifact_root / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    write_run_manifest(artifact_root, manifest["run"])
    print(output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/professional-pilot-v1")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/professional-pilot-v1/professional-pilot-v1-review.html"),
    )
    args = parser.parse_args()
    build_review(args.artifact_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
