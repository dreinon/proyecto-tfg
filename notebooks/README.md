# Notebooks

Use notebooks for exploration, visual analysis, and communicating results. Move reusable data,
model, metric, and plotting logic into `src/score_super_resolution/` and test it there.

Each reported notebook must identify the Git revision, configuration, data/split manifest, seed,
and environment snapshot that produced its results.

## Final SMB comparison

- `03-smb-model-comparison.ipynb` and `phase3-smb-evaluation.zip` are retained as the v1
  development pilot that revealed a scale-transfer defect. Do not use them as final confirmation.
- `03-smb-model-comparison-v2.ipynb` is the final rerun notebook. It validates the frozen staff
  scale of 64 fresh works before inference and writes only to
  `artifacts/phase3-smb-evaluation-v2/` and `phase3-smb-evaluation-v2.zip`.

The rationale and frozen identities are in [`../docs/smb-protocol-v2.md`](../docs/smb-protocol-v2.md).

## Adaptation and professional pilot

- `04-smb-edsr-finetuning.ipynb`: bounded EDSR adaptation with frozen work-disjoint partitions;
  see [the adaptation protocol](../docs/smb-edsr-finetuning-v1.md).
- `05-professional-demonstrator-validation.ipynb`: external professional pilot;
  see [the demonstrator contract](../docs/professional-demonstrator-v1.md).

## Earlier development and review

`smb_human_review.ipynb` and the three `02-*.ipynb` notebooks retain the dataset audit,
degradation calibration and fixture reviews used during development. They are not the entry
point for rerunning the final studies, but they are not unused code either. Keep them and
their generated images as process evidence.
