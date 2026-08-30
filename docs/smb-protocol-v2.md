# SMB final-evaluation protocol v2

## Status and purpose

Protocol v2 corrects a transfer defect found during the first complete SMB run. The degradation
used in v1 had been calibrated on synthetic notation with an approximately 30 px staff spacing,
but its blur was expressed in absolute pixels. The audited v1 SMB pages had smaller and variable
staff scales, so the same nominal `moderate` and `strong` parameters did not preserve their intended
meaning. In particular, some full-page strong cases became effectively destructive.

The correction does not erase that result:

- v1 remains development-pilot and stress-test evidence that exposed the design defect;
- v1 must not be reported as the final model comparison;
- v2 is the only final confirmatory SMB comparison;
- no v2 page, degradation output, reconstruction, or metric was inspected while fixing v2.

The student explicitly approved this corrective rerun on 30 August 2026. The decision and enacted
deviation are recorded as `DEC-SCI-04` and `DEV-SCI-01`.

## Staff-scale normalization

The deterministic estimator `region-deskew-horizontal-morphology-v1` works only from each source
HR page and its official SMB region boxes. It deskews near-horizontal content, extracts long
horizontal components, detects repeated five-line sequences, and freezes the median distance
between adjacent staff lines in HR pixels. It fails closed unless at least two consistent staff
sequences support a value in the validated 4–32 px range.

Before inference, the notebook recomputes all 64 values from the source inputs and requires every
value to agree with the frozen sample within 0.5 px. A failure stops the run before any v2 model
output is generated.

The six final cells preserve x2 and x4 and the three intended roles:

| Severity | Gaussian blur in HR pixels | Noise after reduction | JPEG |
| --- | --- | --- | --- |
| `clean` | none | none | none |
| `moderate` | `max(0.3, 0.05 × staff spacing)` | achromatic Gaussian, σ 1.5 | quality 80, 4:4:4 |
| `strong` | `max(0.3, 0.15 × staff spacing)` | achromatic Gaussian, σ 2.5 | quality 45, 4:4:4 |

All cells reduce with `INTER_AREA`. The master seed is 20260830; every page-condition seed and
effective parameter is retained in a validated trace. The ratios were selected using only the
already-exposed v1 pilot and the earlier accepted synthetic fixture. The final values are slightly
milder than the fixture ratios because the v1 pilot showed the interaction between small SMB staff
scales, x4 reduction, compression, and full-page presentation.

## Fresh confirmatory sample

The v1 CSV labelled page-specific identifiers as source groups. Cross-checking it against the
authoritative 685-row manifest showed that its 64 pages represented 53 independent musical works.
V2 uses the authoritative manifest grouping directly.

The frozen v2 sample contains:

- 64 pages from 64 distinct source works;
- no item used in v1;
- no source work represented in v1;
- one deterministically SHA-ranked estimator-compatible page per selected work;
- staff spacing from 6.526 to 18.275 HR px, with a median of 9.996 px.

Estimator-incompatible pages were skipped by the predeclared input-only rule before the sample was
frozen. This bounds the final claims to fresh SMB pages with a measurable staff scale; it does not
support prevalence claims for every SMB page or uncontrolled real scans.

## Frozen identities

- Sample: `data/audits/smb-evaluation-sample-v2.csv`, SHA-256
  `d2a686c6867a4c5bb3a362d392a466787420e057b217a55cacfe3b7b50fa0523`.
- V1 work exclusion set: `data/audits/smb-evaluation-v1-source-groups.csv`, SHA-256
  `3df562c62bbe718ea2711bdac1eb047a45d58b6e41c4b8236edea433d9b8fb4c`.
- Degradation control: `configs/degradations/staff-scale-score-v2.yaml`, canonical SHA-256
  `253916bcaefb1582556a7f3cf0b57639330092199f35c25c2ce47c24cb4a6e33`.
- Experiment: `configs/experiments/smb-pretrained-evaluation-v2.yaml`.
- Unlock controls: `configs/smb-evaluation-v2/`.
- Notebook: `notebooks/03-smb-model-comparison-v2.ipynb`.
- Output root: `artifacts/phase3-smb-evaluation-v2/`.
- Kaggle archive: `/kaggle/working/phase3-smb-evaluation-v2.zip`.

The notebook must reconcile 1152 method-condition-page rows, 384 unique degradation traces, 64
independent works, six exact conditions, three exact methods, 30 predeclared qualitative PNGs, and
the 64-row preflight before it creates the archive.

## Interpretation

Compare methods only within the same scale and severity. Keep failures and the six qualitative
assignments fixed. Inspect staff continuity, symbol geometry, small-element loss, unintended joins
or separations, text/digit corruption, plausible musical changes, and natural-image texture or
ringing. EDSR or SwinIR texture cannot be removed by tuning on v2; it is a result to document and a
possible motivation for a future adaptation gate using independent training/validation data.
