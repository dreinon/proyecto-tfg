# ruff: noqa: E501
"""Generate the fixed SMB v2 qualitative-review interface."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


class QualitativeReviewError(ValueError):
    """Reject incomplete or inconsistent qualitative evidence."""


METHODS = (
    ("bicubic-opencv-v1", "Bicúbica"),
    ("edsr-baseline-official-v1", "EDSR"),
    ("swinir-lightweight-official-v1", "SwinIR"),
)

TAXONOMY = (
    (
        "staff-line-breakage-or-removal",
        "Rotura o eliminación de líneas de pentagrama",
    ),
    (
        "staff-line-thickening-or-hallucination",
        "Engrosamiento o invención de líneas de pentagrama",
    ),
    ("altered-or-missing-symbol", "Símbolo alterado o ausente"),
    ("unintended-join-or-separation", "Unión o separación indebida"),
    ("text-or-digit-corruption", "Corrupción de texto o dígitos"),
    ("plausible-musical-change", "Cambio musical plausible pero incorrecto"),
    ("natural-image-texture-or-ringing", "Textura artificial, halo o ringing"),
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualitativeReviewError(f"Cannot read {path.name}") from error
    if not isinstance(value, dict):
        raise QualitativeReviewError(f"{path.name} must contain an object")
    return value


def _source_groups(evaluation_root: Path) -> dict[str, str]:
    path = evaluation_root / "staff-scale-preflight.csv"
    try:
        with path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
    except OSError as error:
        raise QualitativeReviewError("Cannot read staff-scale preflight") from error
    groups = {row["item_id"]: row["source_group_id"] for row in rows}
    if len(groups) != 64:
        raise QualitativeReviewError("Staff-scale preflight must identify 64 items")
    return groups


def build_review_spec(
    project_root: Path,
    evaluation_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build and verify the outcome-independent six-case review specification."""

    project_root = project_root.resolve()
    evaluation_root = evaluation_root.resolve()
    output_root = output_root.resolve()
    experiment_path = project_root / "configs/experiments/smb-pretrained-evaluation-v2.yaml"
    try:
        experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise QualitativeReviewError("Cannot read the frozen v2 experiment") from error
    if not isinstance(experiment, dict):
        raise QualitativeReviewError("Frozen v2 experiment must be an object")

    assignment = experiment.get("qualitative", {}).get("assignment")
    conditions = experiment.get("conditions")
    if not isinstance(assignment, list) or len(assignment) != 6:
        raise QualitativeReviewError("Frozen qualitative assignment must contain six cases")
    if not isinstance(conditions, list) or len(conditions) != 6:
        raise QualitativeReviewError("Frozen experiment must contain six conditions")
    assigned_conditions = [row[1] for row in assignment if isinstance(row, list) and len(row) == 2]
    if assigned_conditions != conditions:
        raise QualitativeReviewError(
            "Qualitative assignment must follow the frozen condition order"
        )

    artifact_manifest_path = evaluation_root / "artifact-manifest.json"
    artifact_manifest = _read_json(artifact_manifest_path)
    runtime = _read_json(evaluation_root / "runtime-evidence.json")
    source_groups = _source_groups(evaluation_root)
    cases = []
    expected_images = 0
    for item_id, condition_id in assignment:
        if not isinstance(item_id, str) or not isinstance(condition_id, str):
            raise QualitativeReviewError("Qualitative assignment IDs must be strings")
        case_root = evaluation_root / "qualitative" / item_id / condition_id
        image_names = (
            "reference-hr.png",
            "input-lr-nearest.png",
            *(f"{method_id}.png" for method_id, _ in METHODS),
        )
        images: dict[str, dict[str, Any]] = {}
        dimensions: tuple[int, int] | None = None
        for name in image_names:
            image_path = case_root / name
            relative_to_artifacts = image_path.relative_to(evaluation_root).as_posix()
            evidence = artifact_manifest.get(relative_to_artifacts)
            if not isinstance(evidence, dict) or not image_path.is_file():
                raise QualitativeReviewError(
                    f"Missing frozen qualitative image: {relative_to_artifacts}"
                )
            digest = _sha256_file(image_path)
            if digest != evidence.get("sha256") or image_path.stat().st_size != evidence.get(
                "bytes"
            ):
                raise QualitativeReviewError(
                    f"Qualitative image differs from its manifest: {relative_to_artifacts}"
                )
            with Image.open(image_path) as image:
                current_dimensions = image.size
                image.verify()
            if dimensions is None:
                dimensions = current_dimensions
            elif dimensions != current_dimensions:
                raise QualitativeReviewError(f"Images are not aligned in {item_id}/{condition_id}")
            images[name] = {
                "relative_url": Path(os.path.relpath(image_path, output_root)).as_posix(),
                "sha256": digest,
                "bytes": image_path.stat().st_size,
            }
            expected_images += 1
        if dimensions is None:
            raise QualitativeReviewError("Qualitative case has no images")
        cases.append(
            {
                "item_id": item_id,
                "condition_id": condition_id,
                "source_group_id": source_groups[item_id],
                "scale": int(condition_id[1]),
                "width": dimensions[0],
                "height": dimensions[1],
                "images": images,
            }
        )
    if expected_images != 30:
        raise QualitativeReviewError("The review must resolve exactly 30 frozen PNGs")

    archive_path = evaluation_root.with_suffix(".zip")
    return {
        "schema_version": 1,
        "review_id": "smb-v2-fixed-qualitative-review",
        "experiment_id": experiment.get("experiment_id"),
        "assignment_sha256": experiment.get("qualitative", {}).get("assignment_sha256"),
        "evaluation_bundle_sha256": _sha256_file(archive_path) if archive_path.is_file() else None,
        "evaluation_manifest_sha256": _sha256_file(artifact_manifest_path),
        "evaluation_git_revision": runtime.get("git_revision"),
        "evaluation_recorded_at": runtime.get("recorded_at"),
        "methods": [{"method_id": method_id, "label": label} for method_id, label in METHODS],
        "taxonomy": [{"flag_id": flag_id, "label": label} for flag_id, label in TAXONOMY],
        "cases": cases,
    }


def validate_review_payload(payload: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Validate a completed review against the frozen review specification."""

    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    expected_top_level = {
        "schema_version",
        "review_id",
        "reviewer",
        "started_at",
        "exported_at",
        "complete",
        "evaluation_bundle_sha256",
        "evaluation_manifest_sha256",
        "evaluation_git_revision",
        "assignment_sha256",
        "assessments",
    }
    require(
        set(payload) == expected_top_level,
        "Review contains missing or unknown top-level fields",
    )
    for field in (
        "review_id",
        "evaluation_bundle_sha256",
        "evaluation_manifest_sha256",
        "evaluation_git_revision",
        "assignment_sha256",
    ):
        require(payload.get(field) == spec.get(field), f"Review {field} differs")
    require(payload.get("schema_version") == 1, "Review schema version differs")
    require(payload.get("complete") is True, "Review is not complete")
    reviewer = payload.get("reviewer")
    require(
        isinstance(reviewer, str) and bool(reviewer.strip()) and len(reviewer) <= 200,
        "Reviewer identity is invalid",
    )

    parsed_times: dict[str, datetime] = {}
    for field in ("started_at", "exported_at"):
        value = payload.get(field)
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (AttributeError, TypeError, ValueError):
            errors.append(f"Review {field} is not valid ISO-8601")
            continue
        require(parsed.tzinfo is not None, f"Review {field} lacks a timezone")
        parsed_times[field] = parsed
    if set(parsed_times) == {"started_at", "exported_at"}:
        require(
            parsed_times["exported_at"] >= parsed_times["started_at"],
            "Review was exported before it started",
        )

    assessment_fields = {
        "item_id",
        "condition_id",
        "method_id",
        "status",
        "flags",
        "notes",
    }
    expected = {
        (case["item_id"], case["condition_id"], method["method_id"])
        for case in spec["cases"]
        for method in spec["methods"]
    }
    allowed_flags = {flag["flag_id"] for flag in spec["taxonomy"]}
    allowed_statuses = {"no-clear-issue", "issues-observed"}
    assessments = payload.get("assessments")
    require(isinstance(assessments, list), "Assessments must be a list")
    actual: list[tuple[object, object, object]] = []
    if isinstance(assessments, list):
        require(len(assessments) == len(expected), "Assessment count differs")
        for index, assessment in enumerate(assessments):
            if not isinstance(assessment, dict):
                errors.append(f"Assessment {index} is not an object")
                continue
            require(
                set(assessment) == assessment_fields,
                f"Assessment {index} contains missing or unknown fields",
            )
            actual.append(
                (
                    assessment.get("item_id"),
                    assessment.get("condition_id"),
                    assessment.get("method_id"),
                )
            )
            status = assessment.get("status")
            flags = assessment.get("flags")
            notes = assessment.get("notes")
            require(status in allowed_statuses, f"Assessment {index} status is invalid")
            require(isinstance(flags, list), f"Assessment {index} flags are invalid")
            if isinstance(flags, list):
                require(len(flags) == len(set(flags)), f"Assessment {index} repeats flags")
                require(set(flags) <= allowed_flags, f"Assessment {index} uses unknown flags")
                require(
                    status != "no-clear-issue" or not flags,
                    f"Assessment {index} marks flags without a clear issue",
                )
                require(
                    status != "issues-observed" or bool(flags),
                    f"Assessment {index} reports an issue without flags",
                )
            require(
                isinstance(notes, str) and len(notes) <= 2000,
                f"Assessment {index} notes are invalid",
            )
        require(len(actual) == len(set(actual)), "Assessment identities are duplicated")
        require(set(actual) == expected, "Assessment matrix differs from the frozen assignment")

    if errors:
        raise QualitativeReviewError("; ".join(errors))

    status_counts = Counter(assessment["status"] for assessment in assessments)
    status_by_method = {
        method["method_id"]: dict(
            Counter(
                assessment["status"]
                for assessment in assessments
                if assessment["method_id"] == method["method_id"]
            )
        )
        for method in spec["methods"]
    }
    flags_by_method = {
        method["method_id"]: dict(
            Counter(
                flag
                for assessment in assessments
                if assessment["method_id"] == method["method_id"]
                for flag in assessment["flags"]
            )
        )
        for method in spec["methods"]
    }
    return {
        "schema_version": 1,
        "record_type": "smb-v2-qualitative-review-validation",
        "review_id": payload["review_id"],
        "valid": True,
        "assessment_count": len(assessments),
        "status_counts": dict(status_counts),
        "status_by_method": status_by_method,
        "flags_by_method": flags_by_method,
        "bindings": {
            "evaluation_bundle_sha256": payload["evaluation_bundle_sha256"],
            "evaluation_manifest_sha256": payload["evaluation_manifest_sha256"],
            "evaluation_git_revision": payload["evaluation_git_revision"],
            "assignment_sha256": payload["assignment_sha256"],
        },
    }


def validate_review(
    project_root: Path,
    evaluation_root: Path,
    review_path: Path,
) -> dict[str, Any]:
    """Read and validate a completed review against the local frozen evidence."""

    review_path = review_path.resolve()
    spec = build_review_spec(project_root, evaluation_root, review_path.parent)
    report = validate_review_payload(_read_json(review_path), spec)
    report["review_file"] = review_path.name
    report["review_bytes"] = review_path.stat().st_size
    report["review_sha256"] = _sha256_file(review_path)
    return report


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _context_card(case: dict[str, Any], image_name: str, label: str) -> str:
    image = case["images"][image_name]
    return f"""
      <figure class="context-card">
        <a href="{_e(image["relative_url"])}" target="_blank" rel="noreferrer">
          <img src="{_e(image["relative_url"])}" alt="{_e(label)}" loading="lazy">
        </a>
        <figcaption>{_e(label)} <span>abrir a resolución completa ↗</span></figcaption>
      </figure>"""


def _method_card(
    case: dict[str, Any], method: dict[str, str], taxonomy: list[dict[str, str]]
) -> str:
    item_id = case["item_id"]
    condition_id = case["condition_id"]
    method_id = method["method_id"]
    assessment_id = f"{item_id}|{condition_id}|{method_id}"
    reference = case["images"]["reference-hr.png"]["relative_url"]
    output = case["images"][f"{method_id}.png"]["relative_url"]
    flags = "".join(
        f"""
          <label class="flag">
            <input type="checkbox" data-role="flag" value="{_e(flag["flag_id"])}" disabled>
            <span>{_e(flag["label"])}</span>
          </label>"""
        for flag in taxonomy
    )
    return f"""
      <article class="method-card" data-assessment="{_e(assessment_id)}"
               data-item="{_e(item_id)}" data-condition="{_e(condition_id)}"
               data-method="{_e(method_id)}">
        <header class="method-header">
          <div><span class="eyebrow">Reconstrucción</span><h3>{_e(method["label"])}</h3></div>
          <span class="status-pill" data-role="status-pill">Pendiente</span>
        </header>
        <div class="compare-frame" style="--aspect:{case["width"]} / {case["height"]}">
          <img src="{_e(output)}" alt="Resultado de {_e(method["label"])}" loading="lazy">
          <img class="reference-layer" src="{_e(reference)}" alt="Referencia HR" loading="lazy">
          <div class="split-line" aria-hidden="true"></div>
          <span class="compare-label left">Referencia</span>
          <span class="compare-label right">{_e(method["label"])}</span>
          <input class="split-control" type="range" min="0" max="100" value="50"
                 aria-label="Comparar referencia y {_e(method["label"])}">
        </div>
        <div class="full-links">
          <a href="{_e(reference)}" target="_blank" rel="noreferrer">Referencia completa ↗</a>
          <a href="{_e(output)}" target="_blank" rel="noreferrer">Resultado completo ↗</a>
        </div>
        <fieldset class="decision">
          <legend>Dictamen visual</legend>
          <label class="choice no-issue">
            <input type="radio" name="status-{_e(assessment_id)}" value="no-clear-issue">
            <span>Sin defecto claro</span>
          </label>
          <label class="choice has-issue">
            <input type="radio" name="status-{_e(assessment_id)}" value="issues-observed">
            <span>He observado defectos</span>
          </label>
          <div class="flags" data-role="flags">{flags}</div>
          <label class="notes-label">Observación breve — opcional
            <textarea data-role="notes" rows="3"
              placeholder="Indica dónde aparece el defecto o por qué el caso resulta dudoso."></textarea>
          </label>
        </fieldset>
      </article>"""


def render_review_html(spec: dict[str, Any]) -> str:
    """Render the complete offline review application."""

    cases_html = []
    nav_html = []
    for index, case in enumerate(spec["cases"], start=1):
        anchor = f"case-{index}"
        nav_html.append(
            f'<a href="#{anchor}" data-nav="{anchor}"><span>{index:02d}</span>'
            f"{_e(case['condition_id'])}</a>"
        )
        methods_html = "".join(
            _method_card(case, method, spec["taxonomy"]) for method in spec["methods"]
        )
        cases_html.append(
            f"""
    <section class="case" id="{anchor}" data-case="{_e(case["condition_id"])}">
      <div class="case-heading">
        <div class="case-index">{index:02d}</div>
        <div>
          <span class="eyebrow">Condición congelada · x{case["scale"]}</span>
          <h2>{_e(case["condition_id"])}</h2>
          <p><code>{_e(case["item_id"])}</code> · obra <code>{_e(case["source_group_id"])}</code></p>
        </div>
      </div>
      <div class="context-grid">
        {_context_card(case, "reference-hr.png", "Referencia HR")}
        {_context_card(case, "input-lr-nearest.png", "Entrada LR ampliada con nearest")}
      </div>
      <div class="methods-grid">{methods_html}</div>
    </section>"""
        )

    spec_json = json.dumps(spec, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Revisión cualitativa SMB v2</title>
  <style>
    :root {{
      color-scheme: light;
      --paper:#f2eee4; --paper-deep:#e5ddcf; --ink:#161512; --muted:#696359;
      --rule:#b7ad9c; --card:#fffdf7; --accent:#a53627; --accent-soft:#f1d8cf;
      --ok:#295b47; --pending:#765d20; --shadow:0 18px 50px rgba(42,32,21,.11);
      font-family:"Iowan Old Style","Palatino Linotype",Palatino,serif;
    }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; color:var(--ink); background:
      linear-gradient(90deg,rgba(80,60,35,.035) 1px,transparent 1px) 0 0/28px 28px,
      var(--paper); }}
    button,input,textarea {{ font:inherit; }}
    .masthead {{ padding:58px max(24px,calc((100vw - 1500px)/2)); border-bottom:1px solid var(--rule);
      background:linear-gradient(135deg,#171612 0 62%,#322c23 62%); color:#fbf5e9; }}
    .kicker,.eyebrow {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:.72rem;
      letter-spacing:.14em; text-transform:uppercase; }}
    .masthead h1 {{ max-width:900px; margin:14px 0 12px; font-family:Didot,"Bodoni 72",Georgia,serif;
      font-size:clamp(2.7rem,7vw,6.3rem); font-weight:500; line-height:.9; letter-spacing:-.045em; }}
    .masthead p {{ max-width:780px; color:#d8d0c3; font-size:1.05rem; line-height:1.55; }}
    .protocol {{ max-width:1500px; margin:0 auto; padding:30px 24px; display:grid;
      grid-template-columns:minmax(0,1fr) minmax(260px,420px); gap:28px; }}
    .notice {{ padding:22px 24px; border-left:5px solid var(--accent); background:var(--card); box-shadow:var(--shadow); }}
    .notice strong {{ color:var(--accent); }}
    .controls {{ padding:20px; border:1px solid var(--rule); background:rgba(255,253,247,.72); }}
    .controls label {{ display:block; font-weight:700; margin-bottom:8px; }}
    .controls input[type=text] {{ width:100%; padding:10px 12px; border:1px solid var(--rule); background:white; }}
    .button-row {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
    button,.import-button {{ border:1px solid var(--ink); padding:9px 13px; background:var(--ink); color:white;
      cursor:pointer; text-decoration:none; }}
    button.secondary,.import-button {{ background:transparent; color:var(--ink); }}
    button.danger {{ border-color:var(--accent); color:var(--accent); background:transparent; }}
    .progress-wrap {{ position:sticky; top:0; z-index:10; border-block:1px solid var(--rule);
      background:rgba(242,238,228,.96); backdrop-filter:blur(10px); }}
    .progress-inner {{ max-width:1500px; margin:auto; padding:12px 24px; display:grid;
      grid-template-columns:auto minmax(160px,1fr) auto; gap:14px; align-items:center; }}
    .progress-track {{ height:8px; background:var(--paper-deep); overflow:hidden; }}
    .progress-bar {{ width:0; height:100%; background:var(--accent); transition:width .25s ease; }}
    nav {{ max-width:1500px; margin:0 auto; padding:18px 24px 4px; display:flex; flex-wrap:wrap; gap:8px; }}
    nav a {{ display:flex; gap:8px; align-items:center; padding:8px 11px; color:var(--ink);
      border:1px solid var(--rule); background:rgba(255,253,247,.7); text-decoration:none; }}
    nav a span {{ font-family:ui-monospace,SFMono-Regular,monospace; color:var(--accent); }}
    main {{ max-width:1500px; margin:0 auto; padding:0 24px 100px; }}
    .case {{ padding-top:72px; margin-top:20px; }}
    .case-heading {{ display:grid; grid-template-columns:90px 1fr; align-items:end; border-bottom:3px solid var(--ink); padding-bottom:14px; }}
    .case-index {{ font-family:Didot,"Bodoni 72",serif; font-size:5rem; line-height:.75; color:var(--accent); }}
    h2 {{ font-family:Didot,"Bodoni 72",Georgia,serif; font-size:clamp(2.2rem,5vw,4.5rem); font-weight:500; margin:4px 0; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:.82em; }}
    .context-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin:24px 0 34px; }}
    figure {{ margin:0; }} .context-card {{ padding:12px; background:var(--card); border:1px solid var(--rule); }}
    .context-card img {{ width:100%; height:min(52vh,620px); object-fit:contain; background:white; display:block; }}
    figcaption {{ display:flex; justify-content:space-between; gap:10px; margin-top:9px; font-weight:700; }}
    figcaption span,.full-links {{ color:var(--muted); font-size:.83rem; font-weight:400; }}
    .methods-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:18px; align-items:start; }}
    .method-card {{ background:var(--card); border:1px solid var(--rule); box-shadow:var(--shadow); overflow:hidden; }}
    .method-card.complete {{ border-color:var(--ok); }}
    .method-header {{ padding:16px 18px; display:flex; justify-content:space-between; gap:12px; align-items:center; border-bottom:1px solid var(--rule); }}
    .method-header h3 {{ font-family:Didot,"Bodoni 72",Georgia,serif; font-size:1.65rem; font-weight:500; margin:3px 0 0; }}
    .status-pill {{ padding:5px 8px; color:var(--pending); background:#eee2bd; font:700 .72rem ui-monospace,SFMono-Regular,monospace; text-transform:uppercase; }}
    .complete .status-pill {{ color:var(--ok); background:#dceade; }}
    .compare-frame {{ --split:50%; position:relative; width:100%; aspect-ratio:var(--aspect); max-height:680px; overflow:hidden; background:white; }}
    .compare-frame img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }}
    .compare-frame .reference-layer {{ clip-path:inset(0 calc(100% - var(--split)) 0 0); }}
    .split-line {{ position:absolute; z-index:2; top:0; bottom:0; left:var(--split); width:2px; background:var(--accent); transform:translateX(-1px); pointer-events:none; }}
    .split-control {{ position:absolute; z-index:3; inset:auto 12px 12px; width:calc(100% - 24px); accent-color:var(--accent); }}
    .compare-label {{ position:absolute; z-index:2; top:10px; padding:4px 7px; color:white; background:rgba(22,21,18,.82); font:700 .68rem ui-monospace,SFMono-Regular,monospace; text-transform:uppercase; }}
    .compare-label.left {{ left:10px; }} .compare-label.right {{ right:10px; }}
    .full-links {{ display:flex; justify-content:space-between; gap:10px; padding:10px 14px; border-block:1px solid var(--rule); }}
    .full-links a {{ color:var(--accent); }}
    .decision {{ margin:0; padding:16px; border:0; }} .decision legend {{ padding-top:16px; font-weight:700; }}
    .choice,.flag {{ display:flex; gap:9px; align-items:flex-start; cursor:pointer; }}
    .choice {{ margin:8px 0; padding:9px; border:1px solid var(--rule); }}
    .choice:has(input:checked) {{ border-color:var(--accent); background:var(--accent-soft); }}
    .flags {{ display:grid; gap:7px; margin:12px 0 16px; padding:13px; background:var(--paper); }}
    .flag:has(input:disabled) {{ opacity:.48; cursor:not-allowed; }}
    input[type=checkbox],input[type=radio] {{ margin-top:2px; accent-color:var(--accent); }}
    .notes-label {{ display:grid; gap:6px; color:var(--muted); font-size:.88rem; }}
    textarea {{ width:100%; resize:vertical; padding:9px; color:var(--ink); border:1px solid var(--rule); background:white; }}
    .saved {{ position:fixed; right:18px; bottom:18px; z-index:20; padding:9px 13px; color:white; background:var(--ok); opacity:0; transform:translateY(10px); transition:.2s; }}
    .saved.show {{ opacity:1; transform:none; }}
    footer {{ padding:28px; text-align:center; border-top:1px solid var(--rule); color:var(--muted); }}
    @media(max-width:1100px) {{ .methods-grid {{ grid-template-columns:1fr; }} .compare-frame {{ max-height:760px; }} }}
    @media(max-width:760px) {{ .protocol,.context-grid {{ grid-template-columns:1fr; }} .case-heading {{ grid-template-columns:60px 1fr; }} .case-index {{ font-size:3.5rem; }} .progress-inner {{ grid-template-columns:1fr auto; }} .progress-track {{ grid-column:1/-1; grid-row:2; }} }}
    @media(prefers-reduced-motion:reduce) {{ html {{ scroll-behavior:auto; }} * {{ transition:none!important; }} }}
  </style>
</head>
<body>
  <header class="masthead">
    <span class="kicker">TFG · SMB · protocolo v2</span>
    <h1>Mesa de cotejo<br>cualitativo</h1>
    <p>Seis casos fijados antes de observar resultados. Compara cada reconstrucción con su referencia y conserva tanto los defectos como los resultados negativos.</p>
  </header>
  <section class="protocol">
    <div class="notice">
      <strong>Qué cuenta como defecto.</strong> Marca alteraciones claras de pentagramas, símbolos, uniones, texto o significado musical. Una diferencia de nitidez no es por sí sola un error, salvo que introduzca textura artificial, halos o afecte la notación. Usa los enlaces de resolución completa cuando haya dudas.
    </div>
    <div class="controls">
      <label for="reviewer">Identificador del revisor</label>
      <input id="reviewer" type="text" value="student-reviewer" autocomplete="off">
      <div class="button-row">
        <button id="export" type="button">Descargar revisión JSON</button>
        <label class="import-button">Importar JSON<input id="import" type="file" accept="application/json" hidden></label>
        <button id="reset" class="danger" type="button">Reiniciar</button>
      </div>
    </div>
  </section>
  <div class="progress-wrap">
    <div class="progress-inner"><strong id="progress-label">0 / 18 dictámenes</strong><div class="progress-track"><div class="progress-bar" id="progress-bar"></div></div><span id="autosave">Guardado local automático</span></div>
  </div>
  <nav>{"".join(nav_html)}</nav>
  <main>{"".join(cases_html)}</main>
  <div class="saved" id="saved">Guardado</div>
  <footer>Revisión fija SMB v2 · sin métricas visibles · los métodos y el orden proceden del protocolo congelado</footer>
  <script>
    const SPEC={spec_json};
    const STORAGE_KEY=`${{SPEC.review_id}}:${{SPEC.evaluation_bundle_sha256 || SPEC.evaluation_manifest_sha256}}`;
    const state={{reviewer:"student-reviewer",started_at:new Date().toISOString(),assessments:{{}}}};
    const cards=[...document.querySelectorAll("[data-assessment]")];
    const saved=document.getElementById("saved");

    function assessmentFor(card) {{
      const id=card.dataset.assessment;
      state.assessments[id] ||= {{
        item_id:card.dataset.item, condition_id:card.dataset.condition,
        method_id:card.dataset.method, status:null, flags:[], notes:""
      }};
      return state.assessments[id];
    }}
    function isComplete(value) {{
      return value.status==="no-clear-issue" || (value.status==="issues-observed" && value.flags.length>0);
    }}
    function flashSaved() {{ saved.classList.add("show"); clearTimeout(window.savedTimer); window.savedTimer=setTimeout(()=>saved.classList.remove("show"),900); }}
    function persist(show=true) {{
      state.reviewer=document.getElementById("reviewer").value.trim() || "student-reviewer";
      localStorage.setItem(STORAGE_KEY,JSON.stringify(state)); updateProgress(); if(show) flashSaved();
    }}
    function updateProgress() {{
      const done=cards.filter(card=>isComplete(assessmentFor(card))).length;
      document.getElementById("progress-label").textContent=`${{done}} / ${{cards.length}} dictámenes`;
      document.getElementById("progress-bar").style.width=`${{100*done/cards.length}}%`;
      cards.forEach(card=>{{ const complete=isComplete(assessmentFor(card)); card.classList.toggle("complete",complete); card.querySelector('[data-role="status-pill"]').textContent=complete?"Registrado":"Pendiente"; }});
    }}
    function applyCard(card) {{
      const value=assessmentFor(card);
      card.querySelectorAll('input[type="radio"]').forEach(input=>input.checked=input.value===value.status);
      const enabled=value.status==="issues-observed";
      card.querySelectorAll('[data-role="flag"]').forEach(input=>{{input.disabled=!enabled; input.checked=value.flags.includes(input.value);}});
      card.querySelector('[data-role="notes"]').value=value.notes || "";
    }}
    function restore(payload) {{
      if(!payload || typeof payload!=="object" || typeof payload.assessments!=="object") throw new Error("Formato de revisión no válido");
      state.reviewer=payload.reviewer || "student-reviewer"; state.started_at=payload.started_at || new Date().toISOString(); state.assessments=payload.assessments;
      document.getElementById("reviewer").value=state.reviewer; cards.forEach(applyCard); persist(false);
    }}
    cards.forEach(card=>{{
      assessmentFor(card); applyCard(card);
      card.querySelectorAll('input[type="radio"]').forEach(input=>input.addEventListener("change",()=>{{
        const value=assessmentFor(card); value.status=input.value;
        if(input.value==="no-clear-issue") value.flags=[];
        applyCard(card); persist();
      }}));
      card.querySelectorAll('[data-role="flag"]').forEach(input=>input.addEventListener("change",()=>{{
        const value=assessmentFor(card); value.flags=[...card.querySelectorAll('[data-role="flag"]:checked')].map(node=>node.value); persist();
      }}));
      card.querySelector('[data-role="notes"]').addEventListener("input",event=>{{assessmentFor(card).notes=event.target.value; persist(false);}});
      const slider=card.querySelector(".split-control"); slider.addEventListener("input",()=>card.querySelector(".compare-frame").style.setProperty("--split",`${{slider.value}}%`));
    }});
    document.getElementById("reviewer").addEventListener("input",()=>persist(false));
    document.getElementById("export").addEventListener("click",()=>{{
      persist(false); const assessments=cards.map(card=>assessmentFor(card));
      const payload={{schema_version:1,review_id:SPEC.review_id,reviewer:state.reviewer,started_at:state.started_at,exported_at:new Date().toISOString(),complete:assessments.every(isComplete),evaluation_bundle_sha256:SPEC.evaluation_bundle_sha256,evaluation_manifest_sha256:SPEC.evaluation_manifest_sha256,evaluation_git_revision:SPEC.evaluation_git_revision,assignment_sha256:SPEC.assignment_sha256,assessments}};
      const blob=new Blob([JSON.stringify(payload,null,2)+"\\n"],{{type:"application/json"}}); const link=document.createElement("a"); link.href=URL.createObjectURL(blob); link.download="smb-v2-qualitative-review.json"; link.click(); URL.revokeObjectURL(link.href);
    }});
    document.getElementById("import").addEventListener("change",async event=>{{
      try {{ restore(JSON.parse(await event.target.files[0].text())); }} catch(error) {{ alert(error.message); }} event.target.value="";
    }});
    document.getElementById("reset").addEventListener("click",()=>{{ if(confirm("¿Borrar todos los dictámenes guardados en este navegador?")) {{ localStorage.removeItem(STORAGE_KEY); state.started_at=new Date().toISOString(); state.assessments={{}}; cards.forEach(card=>{{assessmentFor(card);applyCard(card);}}); persist(false); }} }});
    try {{ const stored=localStorage.getItem(STORAGE_KEY); if(stored) restore(JSON.parse(stored)); }} catch(error) {{ console.warn("No se pudo restaurar la revisión",error); }}
    updateProgress();
  </script>
</body>
</html>
"""


def generate_review(
    project_root: Path,
    evaluation_root: Path,
    output_root: Path,
) -> tuple[Path, Path]:
    """Generate the HTML and a content-addressed manifest atomically."""

    output_root.mkdir(parents=True, exist_ok=True)
    spec = build_review_spec(project_root, evaluation_root, output_root)
    rendered = render_review_html(spec).encode("utf-8")
    html_path = output_root / "qualitative-review-v2.html"
    temporary = html_path.with_suffix(".html.tmp")
    temporary.write_bytes(rendered)
    os.replace(temporary, html_path)
    spec_bytes = (json.dumps(spec, sort_keys=True, separators=(",", ":")) + "\n").encode()
    manifest = {
        "schema_version": 1,
        "review_id": spec["review_id"],
        "generated_at": datetime.now(UTC).isoformat(),
        "html_path": html_path.name,
        "html_bytes": len(rendered),
        "html_sha256": _sha256_bytes(rendered),
        "review_spec_sha256": _sha256_bytes(spec_bytes),
        "case_count": len(spec["cases"]),
        "assessment_count": len(spec["cases"]) * len(spec["methods"]),
        "image_count": sum(len(case["images"]) for case in spec["cases"]),
        "evaluation_bundle_sha256": spec["evaluation_bundle_sha256"],
        "evaluation_manifest_sha256": spec["evaluation_manifest_sha256"],
    }
    manifest_path = output_root / "qualitative-review-manifest.json"
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_bytes(manifest_payload)
    os.replace(manifest_tmp, manifest_path)
    return html_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("artifacts/kaggle/phase3-smb-evaluation-v2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/kaggle/phase3-smb-analysis-v2"),
    )
    args = parser.parse_args()
    html_path, manifest_path = generate_review(
        args.project_root,
        args.evaluation_root,
        args.output_root,
    )
    print(json.dumps({"html": str(html_path), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
