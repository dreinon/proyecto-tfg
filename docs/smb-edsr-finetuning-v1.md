# SMB EDSR fine-tuning protocol v1

## Purpose and claim boundary

This is a bounded secondary study of **within-SMB domain adaptation**. It asks whether fine-tuning
the official EDSR baseline on music-score patches improves reconstruction on previously unseen SMB
works relative to the same pretrained checkpoint and bicubic interpolation.

It does not replace the primary 64-work pretrained comparison, create an official SMB split, prove
generalization to real scans or other collections, or validate musical correctness through PSNR or
SSIM alone. SwinIR is not fine-tuned: one architecture at two scales is sufficient to answer the
adaptation question without opening a new model search.

## Relationship with the previous SMB studies

- The final v2 benchmark's 64 source works are excluded from train, validation, and test.
- The v1 pilot was already disclosed as development evidence. Its source works may enter **train
  only**; none may enter validation or test.
- Validation and test use staff-measurable source works that appeared in neither v1 nor v2.
- The official Hugging Face split remains recorded as `test`. The three roles below are
  project-defined partitions over one pinned corpus revision.

This design retains the uncontaminated primary benchmark while permitting a separate adaptation
question. Results from both studies must be reported under their own denominators and may not be
pooled.

## Frozen data partition

The source is `PRAIG/SMB` at revision
`96332e8c4ac81cbdb7f61093ec5a4bfff76a0adb`. The partition is frozen before training in
`data/adaptation/smb-edsr-finetuning-v1-split.csv` with SHA-256
`ee3e2834679a184168e9fe689eb3e9575d450dbd21be36090eebdb544477075f`.

| Partition | Prior role | Works | Eligible pages | Pages used for selection/evaluation |
|---|---|---:|---:|---:|
| Train | v1 development only | 45 | 212 | patches sampled from all eligible pages |
| Validation | fresh holdout | 13 | 35 | 13 fixed representative pages |
| Test | fresh holdout | 20 | 55 | 20 fixed representative pages |

The unit of separation and statistical inference is `source_group_id`, representing an independent
score/work. Pages and patches are nested observations. Source groups cannot cross partitions.

Eligibility uses only audited input identity, paired-reference status, valid region annotations,
and an input-side staff-spacing estimate. The 134 rejected candidate pages and their reasons are
retained in `data/adaptation/smb-edsr-finetuning-v1-exclusions.csv`; no SR output or metric informed
eligibility. One representative page per work is selected by seeded SHA-256 ranking.

Fresh works are ranked with seed `20260901`; the first 20 form test and the remaining 13 form
validation. Runtime access must reproduce the audited canonical pixel hash for every page used.

## Training contract

Two independent models are initialized from the official checksummed EDSR x2 and x4 checkpoints.
Each is fine-tuned with:

- RGB L1 objective;
- Adam optimizer;
- learning rate `5e-5`, cosine-decayed to `1e-6`;
- at most 2,500 optimization steps per scale;
- batch size 8 and 48x48 LR patches;
- deterministic seed `20260902`;
- mixed float16 on CUDA and float32 otherwise;
- gradient clipping at norm 1.0;
- validation every 250 steps, patience 4 and minimum improvement `1e-5`.

Training samples source works uniformly, then pages within a work. Crops are centred on annotated
notation regions using input/annotation information only. Each batch balances the three frozen
degradation profiles. No flips or rotations that could alter musical geometry are used.

Validation uses one fixed patch for each profile on the representative page of each validation
work. The selected checkpoint minimizes mean validation RGB L1. A step-zero selection is valid if
fine-tuning never improves the pretrained initialization.

## Test and analysis contract

The test partition is not accessed until compatible selected checkpoints exist for both scales.
Each of its 20 representative pages is evaluated under six conditions: x2/x4 crossed with clean,
moderate, and strong. Every case compares:

1. bicubic OpenCV interpolation;
2. the official pretrained EDSR checkpoint;
3. the selected SMB-adapted EDSR checkpoint.

Raw PSNR-Y, SSIM-Y, PSNR-RGB, SSIM-RGB, runtime, output hash, and checkpoint hash are retained for
all 360 tuples. Reporting uses condition-specific means and paired source-level deltas. The primary
adaptation comparison is fine-tuned minus pretrained EDSR; fine-tuned minus bicubic is secondary.
Two-sided percentile bootstrap intervals use 2,000 source resamples and seed `20260903`. Intervals
describe uncertainty across the 20 sampled works, not across pages or pixels.

## Fixed qualitative evidence

Six distinct test works, one per condition, are fixed by seeded identity ranking before training.
For each case the evidence bundle contains the aligned HR reference, LR observation enlarged with
nearest-neighbour interpolation, bicubic output, pretrained EDSR output, and adapted EDSR output.
The review must distinguish defects inherited from LR/degradation from alterations introduced by a
reconstruction. Positive and negative findings are retained under the existing notation-failure
taxonomy.

## Reproducibility and stopping rules

`notebooks/04-smb-edsr-finetuning.ipynb` is the sole accelerated orchestration path. It records the
Git revision, split and source hashes, runtime/GPU/CUDA/PyTorch identity, dependency freeze, seeds,
histories, selected checkpoints, raw and aggregate results, qualitative files, and a checksummed
artifact manifest. Partial evaluation resumes only when its split, conditions, and checkpoint
hashes match exactly.

The study is complete only if all 360 quantitative tuples, 24 paired bootstrap rows, 30 fixed
qualitative PNGs, two selected checkpoints, and both pixel-identity preflights reconcile. Any
failure remains evidence; it is not repaired by inspecting test outcomes or changing the frozen
protocol. Further tuning, model families, realistic-scan evaluation, OMR, or deployment require a
new prospective decision and cannot delay the thesis core.

## Completed-study pointer

The frozen study completed and passed independent reconciliation on 1 September 2026. Validated
outcomes, evidence identities, and claim boundaries are reported separately in
`docs/smb-edsr-finetuning-v1-results.md` so this preregistered protocol remains an auditable record
of what was decided before training and test access.
