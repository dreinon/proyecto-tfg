# Development environment

The project uses `uv` and Python 3.12.12. The lockfile is the dependency source of truth.

## Local CPU setup

```bash
uv sync --extra cpu --group dev --group notebooks --group kaggle
uv run python -m score_super_resolution.environment
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Use `uv add <package>` or `uv add --group <group> <package>` for a justified dependency. Do not
use bare `pip` or edit `uv.lock` manually.

## Kaggle acceleration

Kaggle supplies the GPU runtime and accelerator-aware PyTorch build. Do not install the local
`cpu` extra in a Kaggle notebook. Instead, record the runtime first:

```python
from score_super_resolution.environment import environment_snapshot

environment_snapshot()
```

Then install only project packages missing from the image. Each reported run must retain the
notebook/kernel revision, accelerator, environment snapshot, project Git revision, configuration,
data manifest, seeds, logs, metrics, and exported artefacts.

Never upload or commit `kaggle.json`.

For SMB, accept the dataset conditions with the relevant Hugging Face account and expose the token
through the local environment or Kaggle Secrets as `HF_TOKEN`. Never paste it into source code or
a notebook. Load the exact revision recorded in `data/sources/smb.yaml` so local and Kaggle runs
address the same dataset state.

## Repository layout

- `src/score_super_resolution/`: reusable implementation.
- `tests/`: deterministic unit and integration tests on tiny fixtures.
- `notebooks/`: exploration and presentation; reusable logic belongs in `src/`.
- `configs/`: versioned experiment configuration.
- `data/sources/`: immutable external-source descriptors.
- `data/manifests/`: reproducible selections and project-defined splits.
- `artifacts/`: documentation of artefact conventions; generated outputs are ignored.

## Professional demonstrator

After restoring the validated x2/x4 adaptation checkpoints under
`artifacts/kaggle/smb-edsr-finetuning-v1/training/`, launch the local image demonstrator with:

```bash
uv run python app.py
```

The interface does not need Hugging Face access because it performs inference on a user-supplied
image. Set `SCORE_SR_CHECKPOINT_DIR` only when the private checkpoints are mounted elsewhere. The
full application and external-pilot contract is in
[`docs/professional-demonstrator-v1.md`](docs/professional-demonstrator-v1.md).
