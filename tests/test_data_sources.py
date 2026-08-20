from pathlib import Path

import yaml


def test_smb_descriptor_pins_evaluation_source() -> None:
    path = Path(__file__).parents[1] / "data" / "sources" / "smb.yaml"
    descriptor = yaml.safe_load(path.read_text(encoding="utf-8"))

    revision = descriptor["revision"]
    assert descriptor["repository_id"] == "PRAIG/SMB"
    assert descriptor["role"] == "evaluation_benchmark"
    assert descriptor["upstream_splits"] == {"test": {"examples": 685}}
    assert len(revision) == 40
    assert all(character in "0123456789abcdef" for character in revision)
    assert descriptor["access"]["store_credentials_in_repository"] is False
