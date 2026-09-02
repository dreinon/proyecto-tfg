# SMB EDSR fine-tuning v1: validated results

## Status and evidence identity

The bounded within-SMB adaptation study completed on 1 September 2026 and passed the independent
local reconciliation in `scripts/validate_smb_edsr_finetuning.py`.

- Archive: `artifacts/kaggle/smb-edsr-finetuning-v1.zip`
- Archive SHA-256: `414613156e71e4242f295b68143950122072fd02027600b44337d00ee972cd1f`
- Artifact-manifest SHA-256: `5ae317ea41666d00626fe5fd6c665733a912816d8fde664ffca00817be680236`
- Executed Git revision: `4e7806733d9dd8899d62a3cfca3548ede03d1243`
- SMB revision: `96332e8c4ac81cbdb7f61093ec5a4bfff76a0adb`
- Frozen split SHA-256: `ee3e2834679a184168e9fe689eb3e9575d450dbd21be36090eebdb544477075f`
- Validation assessment: `ready-with-declared-scope`

The corrected runtime record reconciles an initially captured stale revision
(`28d18966a10356183e500b47b72ed35d81dd3e03`) with the clean checkout actually executed
(`4e7806733d9dd8899d62a3cfca3548ede03d1243`). Only the runtime evidence and its enclosing manifest
changed; all metrics, images, checkpoints, histories, and frozen inputs are byte-identical to the
first recovered bundle.

## Reconciliation

All expected evidence is present and independently checked:

| Check | Result |
|---|---:|
| Manifest files | 50 / 50 valid hashes and sizes |
| Source groups | 45 train, 13 validation, 20 test |
| Eligible pages | 212 train, 35 validation, 55 test |
| Cross-partition source overlap | 0 |
| Raw test tuples | 360 / 360 unique |
| Paired bootstrap rows | 24 / 24 reproduced exactly |
| Fixed qualitative PNGs | 30 / 30 pixel hashes valid |
| Selected checkpoints | 2 / 2 identities valid |

The primary 64-work pretrained benchmark is excluded from all three adaptation roles. The roles
are project-defined partitions of SMB's single official `test` split; they are not official SMB
splits and must not be pooled with the primary study.

## Checkpoint selection

Both models were initialized from the official EDSR baseline and selected using validation RGB L1
only. Test data remained locked until compatible checkpoints existed for both scales.

| Scale | Selected step | Completed step | Validation RGB L1 | Interpretation |
|---|---:|---:|---:|---|
| x2 | 1,500 | 2,500 | 0.014086 | Later steps did not improve the frozen selection criterion |
| x4 | 2,500 | 2,500 | 0.031575 | Best point coincided with the frozen budget boundary |

The x4 curve was still improving at step 2,500. This is a limitation of the bounded study, not a
reason to extend training after seeing the result.

## Quantitative outcome

The primary comparison is adapted minus pretrained EDSR on 20 independent test works. All twelve
condition-specific intervals exclude zero.

| Condition | PSNR-Y delta, dB [95% interval] | SSIM-Y delta [95% interval] | Sources improved in PSNR-Y |
|---|---:|---:|---:|
| x2 clean | 3.955 [3.492, 4.373] | 0.0169 [0.0143, 0.0200] | 20 / 20 |
| x2 moderate | 4.719 [4.106, 5.336] | 0.0387 [0.0361, 0.0415] | 20 / 20 |
| x2 strong | 7.844 [7.328, 8.292] | 0.1949 [0.1786, 0.2104] | 20 / 20 |
| x4 clean | 3.599 [2.978, 4.186] | 0.0602 [0.0487, 0.0722] | 20 / 20 |
| x4 moderate | 3.242 [2.647, 3.771] | 0.0796 [0.0660, 0.0933] | 19 / 20 |
| x4 strong | 3.703 [2.723, 4.690] | 0.1558 [0.1365, 0.1746] | 18 / 20 |

The adapted model also exceeds bicubic on both primary metrics in every condition at the aggregate
paired-interval level. Its mean PSNR-Y advantage over bicubic ranges from 4.962 to 8.421 dB.

The two repeated x4 PSNR-Y exceptions versus pretrained EDSR are the pages with the smallest staff
spacing in the 20-work test sample. Because this is a post-hoc observation involving two works, it
is retained only as a limitation and hypothesis for future stratified evaluation.

## Qualitative boundary

The fixed evidence contains one test work per condition and five views per work: HR, enlarged LR,
bicubic, pretrained EDSR, and adapted EDSR. The student completed the notation-level review on
2 September 2026. Four cases (x2/x4 clean and moderate) were classified as improvement without a
clear new defect; the x2/x4 strong cases were classified as damage already present in LR and not
recovered. No case was classified as damage introduced or amplified by adapted EDSR.

The adapted outputs reduce halos and natural-image-like texture and improve the definition of staff
lines and notation. The clean and moderate cases remain understandable despite minor text or
closely spaced notehead limitations. Under strong degradation, small accidentals, ornaments,
ledger lines, fingerings, hollow noteheads, text, and digits can remain lost or malformed. The x4
strong case may support approximate visual consultation, but not final musical interpretation.
These six cases detect failure modes and do not estimate their prevalence. Higher PSNR-Y or SSIM-Y
therefore does not establish musical correctness.

## Defensible conclusion

Within this pinned SMB revision, the project-defined source-disjoint roles, the matched synthetic
degradation protocol, and the frozen 2,500-step budget, bounded EDSR fine-tuning produces a large
and stable fidelity improvement over the official pretrained EDSR checkpoint. This is evidence for
within-corpus domain adaptation, not for real-scan restoration, other collections, OMR accuracy,
archival replacement, autonomous publication, or deployment.

The primary pretrained benchmark remains the basis for comparing EDSR, SwinIR, and bicubic without
SMB adaptation. The adaptation study answers a separate secondary question and does not
retroactively alter model selection in that benchmark.

SMB is declared CC BY-NC 4.0 at dataset level, while per-item provenance is unresolved. The
adapted checkpoints and qualitative SMB images are therefore retained outside Git and must not be
redistributed or used commercially without a separate rights review.
