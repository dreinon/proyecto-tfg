# Project repository instructions

This repository contains the implementation and reproducible experiments for the TFG.

Before substantive work, read `../AGENTS.md`, `docs/research-protocol.md`, and the relevant
academic document under `../memoria/docs/tfg-guidance/`. The parent file defines the source
hierarchy, Overleaf relationship, GSD rules, Git safety, and required verification.

Use `uv` and the checked-in `uv.lock`. Local work is Python 3.12.12 with the `cpu` extra; Kaggle is
the planned GPU environment. Keep reusable logic in `src/`, deterministic tests in `tests/`, small
source descriptors and manifests in `data/`, and exploratory notebooks in `notebooks/`. SMB is a
manually gated Hugging Face evaluation benchmark pinned in `data/sources/smb.yaml`; do not use its
official `test` split for tuning or training without an explicit protocol decision.

Do not commit datasets, generated images, run directories, checkpoints, credentials, or secrets.
Run Ruff and pytest before presenting an implementation as complete.
