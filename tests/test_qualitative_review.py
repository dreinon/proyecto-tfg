from __future__ import annotations

import pytest

from score_super_resolution.qualitative_review import (
    METHODS,
    TAXONOMY,
    QualitativeReviewError,
    render_review_html,
    validate_review_payload,
)


def _spec() -> dict[str, object]:
    cases = []
    for index, condition in enumerate(
        ("x2-clean", "x2-moderate", "x2-strong", "x4-clean", "x4-moderate", "x4-strong")
    ):
        images = {
            name: {"relative_url": f"../evaluation/{index}/{name}", "sha256": "a" * 64}
            for name in (
                "reference-hr.png",
                "input-lr-nearest.png",
                *(f"{method_id}.png" for method_id, _ in METHODS),
            )
        }
        cases.append(
            {
                "item_id": f"smb-test-{index:06d}",
                "condition_id": condition,
                "source_group_id": f"work-{index}",
                "scale": int(condition[1]),
                "width": 1200,
                "height": 1600,
                "images": images,
            }
        )
    return {
        "schema_version": 1,
        "review_id": "smb-v2-fixed-qualitative-review",
        "experiment_id": "smb-pretrained-evaluation-v2",
        "assignment_sha256": "b" * 64,
        "evaluation_bundle_sha256": "c" * 64,
        "evaluation_manifest_sha256": "d" * 64,
        "evaluation_git_revision": "e" * 40,
        "evaluation_recorded_at": "2026-08-31T00:00:00+00:00",
        "methods": [{"method_id": method_id, "label": label} for method_id, label in METHODS],
        "taxonomy": [{"flag_id": flag_id, "label": label} for flag_id, label in TAXONOMY],
        "cases": cases,
    }


def test_review_html_contains_the_fixed_cases_methods_and_taxonomy() -> None:
    rendered = render_review_html(_spec())

    assert rendered.count('class="case"') == 6
    assert rendered.count('class="method-card"') == 18
    assert rendered.count('type="checkbox" data-role="flag"') == 18 * len(TAXONOMY)
    for method_id, label in METHODS:
        assert method_id in rendered
        assert label in rendered
    for flag_id, label in TAXONOMY:
        assert flag_id in rendered
        assert label in rendered


def test_review_html_persists_and_exports_attributable_decisions_without_metrics() -> None:
    rendered = render_review_html(_spec())

    assert "localStorage.setItem" in rendered
    assert 'link.download="smb-v2-qualitative-review.json"' in rendered
    assert "evaluation_bundle_sha256" in rendered
    assert "assignment_sha256" in rendered
    assert "psnr" not in rendered.lower()
    assert "ssim" not in rendered.lower()
    assert "Sin defecto claro" in rendered
    assert "He observado defectos" in rendered


def _completed_review() -> dict[str, object]:
    spec = _spec()
    assessments = []
    for case in spec["cases"]:
        for method in spec["methods"]:
            assessments.append(
                {
                    "item_id": case["item_id"],
                    "condition_id": case["condition_id"],
                    "method_id": method["method_id"],
                    "status": "no-clear-issue",
                    "flags": [],
                    "notes": "",
                }
            )
    return {
        "schema_version": 1,
        "review_id": spec["review_id"],
        "reviewer": "student-reviewer",
        "started_at": "2026-08-31T10:00:00Z",
        "exported_at": "2026-08-31T11:00:00Z",
        "complete": True,
        "evaluation_bundle_sha256": spec["evaluation_bundle_sha256"],
        "evaluation_manifest_sha256": spec["evaluation_manifest_sha256"],
        "evaluation_git_revision": spec["evaluation_git_revision"],
        "assignment_sha256": spec["assignment_sha256"],
        "assessments": assessments,
    }


def test_completed_review_is_bound_to_every_frozen_case_and_method() -> None:
    report = validate_review_payload(_completed_review(), _spec())

    assert report["valid"] is True
    assert report["assessment_count"] == 18
    assert report["status_counts"] == {"no-clear-issue": 18}


def test_review_rejects_issue_without_a_taxonomy_flag() -> None:
    review = _completed_review()
    review["assessments"][0]["status"] = "issues-observed"

    with pytest.raises(QualitativeReviewError, match="without flags"):
        validate_review_payload(review, _spec())
