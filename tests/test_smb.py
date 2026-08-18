from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from score_super_resolution.smb import check_access, load_smb, main

from score_super_resolution.benchmark_policy import BenchmarkPolicyError, BenchmarkPurpose

EXPECTED_REVISION = "96332e8c4ac81cbdb7f61093ec5a4bfff76a0adb"


def test_load_smb_guards_before_using_exact_descriptor_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    dataset = object()

    def fake_guard(**kwargs: Any) -> object:
        events.append(("guard", kwargs["purpose"]))
        return kwargs["callback"]()

    def fake_loader(repository_id: str, *, split: str, revision: str) -> object:
        events.append(("loader", (repository_id, split, revision)))
        return dataset

    monkeypatch.setattr("score_super_resolution.smb.assert_smb_purpose_allowed", fake_guard)

    loaded = load_smb(purpose=BenchmarkPurpose.CONTENT_AUDIT, loader=fake_loader)

    assert loaded is dataset
    assert events == [
        ("guard", BenchmarkPurpose.CONTENT_AUDIT),
        ("loader", ("PRAIG/SMB", "test", EXPECTED_REVISION)),
    ]
    assert len(EXPECTED_REVISION) == 40


def test_load_smb_policy_rejection_prevents_loader_side_effect() -> None:
    calls: list[str] = []

    with pytest.raises(BenchmarkPolicyError, match="inference"):
        load_smb(
            purpose=BenchmarkPurpose.INFERENCE,
            loader=lambda *_args, **_kwargs: calls.append("loaded"),
        )

    assert calls == []


def test_metadata_access_reports_allowlisted_identity_schema_and_counts_only() -> None:
    class MetadataOnlyBuilder:
        info = SimpleNamespace(
            features={
                "image": object(),
                "id": object(),
                "original_width": object(),
                "original_height": object(),
                "regions": object(),
                "page_texture": object(),
                "page": object(),
                "original_score": object(),
            },
            splits={"test": SimpleNamespace(num_examples=685)},
        )

        def download_and_prepare(self) -> None:
            raise AssertionError("metadata access must not download the payload")

        def as_dataset(self) -> None:
            raise AssertionError("metadata access must not decode the payload")

    calls: list[tuple[str, str]] = []

    def fake_builder_loader(repository_id: str, *, revision: str) -> MetadataOnlyBuilder:
        calls.append((repository_id, revision))
        return MetadataOnlyBuilder()

    diagnostic = check_access(builder_loader=fake_builder_loader)

    assert calls == [("PRAIG/SMB", EXPECTED_REVISION)]
    assert diagnostic == {
        "status": "accessible",
        "repository_id": "PRAIG/SMB",
        "resolved_revision": EXPECTED_REVISION,
        "features": [
            "id",
            "image",
            "original_height",
            "original_score",
            "original_width",
            "page",
            "page_texture",
            "regions",
        ],
        "splits": {"test": {"examples": 685}},
        "descriptor_match": True,
    }


@pytest.mark.parametrize(
    "unsafe_detail",
    (
        "credential-sentinel-never-print",
        "Authorization: Bearer credential-sentinel-never-print",
        "https://user:credential-sentinel-never-print@example.invalid/data?token=secret",
    ),
)
def test_access_diagnostics_never_reproduce_exception_details(
    unsafe_detail: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def rejected_builder(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(unsafe_detail)

    diagnostic = check_access(builder_loader=rejected_builder)
    monkeypatch.setattr("score_super_resolution.smb.check_access", lambda: diagnostic)

    exit_code = main(["check-access", "--metadata-only"])
    captured = capsys.readouterr()
    rendered = json.dumps(diagnostic) + captured.out + captured.err

    assert exit_code == 2
    assert diagnostic == {
        "status": "blocked",
        "reason": "SMB metadata access unavailable",
        "action": "Confirm approved Hugging Face access and cached authentication, then retry.",
    }
    assert unsafe_detail not in rendered
    assert "credential-sentinel-never-print" not in rendered
    assert "Authorization:" not in rendered
    assert "Bearer " not in rendered
    assert "user:credential" not in rendered
    assert "?token=" not in rendered


def test_public_access_api_has_no_token_parameter() -> None:
    for function in (load_smb, check_access):
        parameters = inspect.signature(function).parameters
        assert "token" not in parameters
        assert all("credential" not in name for name in parameters)


def test_cli_requires_explicit_metadata_only_mode() -> None:
    with pytest.raises(SystemExit) as error:
        main(["check-access"])

    assert error.value.code == 2


@pytest.mark.skipif(
    os.environ.get("RUN_SMB_METADATA_INTEGRATION") != "1",
    reason="authenticated SMB metadata check is opt-in",
)
def test_authenticated_smb_metadata_access() -> None:
    diagnostic = check_access()

    assert diagnostic["status"] == "accessible", diagnostic
    assert diagnostic["repository_id"] == "PRAIG/SMB"
    assert diagnostic["resolved_revision"] == EXPECTED_REVISION
    assert diagnostic["descriptor_match"] is True
    assert diagnostic["splits"] == {"test": {"examples": 685}}


def test_default_suite_keeps_network_integration_opt_in() -> None:
    source = Path(__file__).read_text(encoding="utf-8")

    assert 'os.environ.get("RUN_SMB_METADATA_INTEGRATION") != "1"' in source
    assert "HF_TOKEN" not in source
