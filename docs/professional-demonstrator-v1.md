# Professional demonstrator and external applicability pilot v1

Status: implementation and external evaluation complete; qualitative review and institutional
agreement pending

## Professional purpose

The extension demonstrates one bounded workflow: create a reversible enlarged consultation copy
from a low-resolution score image, compare it with the source, retain model identity, and let the
operator accept or reject the derivative. It is not an automatic restoration product and does not
authorize editorial, archival, OMR, or commercial use.

The application is image-only and runs the validated SMB-adapted EDSR checkpoints at x2 or x4. It
does not ingest PDF, create accounts, retain uploads, overwrite originals, or conceal limitations.

## Application contract

The interface must:

1. accept one bounded RGB PNG/JPEG image;
2. require an explicit x2 or x4 scale;
3. show the original and derivative together at inspectable zoom;
4. expose a downloadable PNG derivative;
5. report runtime, method, device, checkpoint identity, and output identity;
6. state that lost symbols, text, digits, and ornaments may not be recoverable;
7. reject unsafe image sizes and unsupported inputs;
8. process one request at a time by default and keep no project-owned copy of the upload.

Gradio may use transient framework-managed files while serving a request; the local application
keeps no history or database and configures hourly temporary-cache cleanup. This is not a promise
about a future hosting provider, whose storage and logging would require a separate deployment
review.

The local entry point is:

```bash
uv run python app.py
```

By default the app resolves the two validated checkpoints under
`artifacts/kaggle/smb-edsr-finetuning-v1/training/`. A private deployment may instead set
`SCORE_SR_CHECKPOINT_DIR` to a mounted, non-public checkpoint directory. The checkpoint files must
not be committed or placed in a public container image before the rights review is closed.

## External input handoff

Place candidate pages under the ignored directory:

```text
data/raw/professional-pilot-v1/
```

Use one lossless PNG or TIFF page per independent work where possible. JPEG is accepted for an
existing source but should not be introduced merely for transfer. Filenames should be stable,
non-sensitive identifiers such as `work-001.png`; bibliographic title, provenance, source type,
rights basis, and any public URL belong in the later manifest rather than the filename.

Raw pages, private collection information, and unlicensed images remain outside Git. A page may
appear in the thesis or public application only when its reproduction rights are explicit.

## Outcome-blind selection

- Reserve 3--5 works for loader and estimator engineering only; they never enter reported results.
- Freeze exactly 12 different works for the professional pilot, with one representative page per
  work and no overlap with SMB.
- Select from input-only properties before opening any SR result: source type, engraving or scan,
  notation density, staff size, page condition, text/digit presence, and rights.
- Record every candidate and exclusion. Estimator or format failure is an exclusion; poor model
  output is not.
- Treat the work as the independent unit. Additional pages may illustrate a use case but may not
  inflate the denominator.

### Provenance and processing boundary

The pages come from the music archive of the Societat Joventut Musical d'Albal (SJMA). The student
accessed them with the entity's authorization while also serving as its president and legal
representative. This dual role and the exact institutional provenance must remain explicit: the
student compiled the evaluation corpus but does not own the underlying scores, arrangements, or
editions.

All pages remain `private-study-only` while the UPV--SJMA agreement for the punctual provision of
materials is being processed. Full source, LR, and SR pages may not enter Git, the public
demonstrator, the thesis, or defence slides. Aggregate evidence may be promoted only after the
administrative basis is resolved. Small analytical excerpts would additionally require a
case-specific copyright and attribution decision; the institutional agreement does not transfer
third-party rights.

The default execution venue for this private corpus is the local project environment. On 3
September 2026, the student, acting as SJMA president and legal representative, confirmed that the
entity authorizes uploading the supplied files to whichever compute service is selected for this
work. This permits Kaggle or another remote environment as a fallback for the experiment. That
processing authorization is not, by itself, permission to publish a public dataset, public notebook
outputs, or the score pages: those acts additionally depend on the rights in each underlying work,
arrangement, and edition. The local CPU path remains preferred while it completes within the
available review window.

### Frozen input selection (3 September 2026)

The sixteen one-page score parts supplied from the SJMA archive were inspected only as HR inputs.
Before any LR, SR, metric, or model output was generated, three estimator failures were assigned to
engineering:
*Caridad del Guadalquivir* (horizontal trumpet scan), *Quadres d'una exposició* (vertical flute
scan), and *Saga Candida* (vertical baritone part). They motivated the input-only projection
fallback in `full-page-hybrid-horizontal-v2` and never enter reported results.

The twelve test works are *Capitania Cides*, *City in Three Words*, *Cullera Suite*, *El jardín de
Hera*, *How to Train Your Dragon*, *La rosa i el drac*, *Lorencín Mendoza*, *Malaguenya de
Barxeta*, *Mari Carmen*, *Theme from Jurassic Park*, *Three Revelations from the Lotus Sutra*, and
*Xàtiva 1939: El Guernica valencià*. This set retains four horizontal and eight vertical parts,
pitched and unpitched percussion, piano, woodwind, saxophone, and low/high brass, plus sparse,
medium, and dense notation across several genres.

*El príncipe de Egipto* is the sole reserve because its vertical film-music/saxophone properties
are already represented by the selected film-music and saxophone cases. This is an input-only
redundancy decision, not an output-quality exclusion. All selected files are provisionally marked
`private-study-only`; no page image may be reproduced publicly unless its rights basis is replaced
with verified permission.

The resulting 15-row input manifest was frozen before generating any LR or SR output with SHA-256
`5e5a1a1be2ea73fc4e65795165acb40c488fa4003eb5b72e5b17ab56cbcca126`. This supersedes the
pre-provenance digest without changing membership or selection: only the source reference was
corrected from a generic student-copy label to the SJMA archive. Its full-page staff estimates
range from 10 to 15 HR pixels.

## Controlled external test

Each frozen HR page remains immutable and acts as the aligned reference. Generate the six existing
conditions (x2/x4 by clean/moderate/strong) with the frozen staff-relative degradation parameters,
then run bicubic, official EDSR, and adapted EDSR. The external pages are test-only: they may not
change degradation parameters, checkpoint selection, model weights, thresholds, or qualitative
sampling.

The design yields 72 page-condition inputs and 216 method outputs. Report PSNR-Y and SSIM-Y,
paired adapted-minus-official deltas, per-condition denominators, runtime, and input/output hashes.
Aggregate and resample by work. The primary professional question is whether the adapted model and
interface produce inspectable consultation derivatives on external material, not whether twelve
works establish universal generalization.

The run must also retain `evaluation-identity.json`, `runtime-evidence.json`, and the checksummed
`artifact-manifest.json`. Resumption is accepted only when the input manifest, degradation,
checkpoints, source bytes, Git revision, device, and fixed qualitative assignment still match.

This remains an external-corpus test with synthetic LR. Even when HR pages are genuine scans with
paper or printing defects, it is not evidence about naturally acquired low-resolution pairs.

### Executed evidence (3 September 2026)

The local CPU run completed from clean Git revision
`62db40ff8f9491f244ddfaeb0cdc13f7482d8718` in 3396.28 seconds. It reconciled all 216 unique
work-condition-method outputs across twelve independent works, six conditions, and three methods;
the analysis produced eighteen aggregate rows, twenty-four paired bootstrap rows, and sixty images
for the twelve fixed qualitative cases. The artifact manifest verifies 68 retained files. Raw
pages, generated page images, checkpoints, and the evidence bundle remain ignored by Git.

Before qualitative interpretation, the adapted EDSR has positive mean adapted-minus-official
deltas for both PSNR-Y and SSIM-Y in every condition. The PSNR-Y differences range from +1.838 dB
(`x4-moderate`) to +8.204 dB (`x2-strong`); the SSIM-Y differences range from +0.0050
(`x2-clean`) to +0.1581 (`x2-strong`). The source-level 95% bootstrap interval excludes zero for
every adapted-versus-official comparison except SSIM-Y under `x2-clean`. These are fidelity results
against synthetically degraded external HR pages, not yet evidence that every derivative is
professionally acceptable. The fixed twelve-case human review remains mandatory.

## Qualitative and operational acceptance

Freeze twelve review cases from input-only strata before opening method outputs. Review HR, LR,
bicubic, official EDSR, and adapted EDSR together using the existing music-notation taxonomy. In
addition, record:

- whether the app completed the page without manual intervention;
- runtime and output dimensions;
- whether original/derivative comparison and download work;
- whether the derivative is acceptable for consultation, acceptable only with reservations, or
  rejected;
- the concrete musical or document defect supporting that decision.

No success percentage or usability claim may be reported from an unexecuted notebook or from a
single operator beyond this bounded walkthrough.

After the run, the notebook generates
`artifacts/professional-pilot-v1/professional-pilot-v1-review.html` and a downloadable ZIP that
contains the HTML, its relative images, metrics, and identities. Extract the complete package,
open the HTML in a browser, complete all twelve cases, and save the downloaded
`professional-pilot-v1-qualitative-review.json` inside that same artifact directory. Validate it
with:

```bash
uv run python scripts/validate_professional_pilot_review.py \
  artifacts/professional-pilot-v1/professional-pilot-v1-qualitative-review.json
```

The validator writes `qualitative-review-validation.json` and refreshes the checksummed artifact
manifest so the reviewed decisions are part of the final evidence bundle.

## Completion and stop rules

Promote the extension into the thesis only if all twelve works, 216 outputs, metrics, identities,
fixed qualitative cases, and application checks reconcile and the final Overleaf candidate can be
reviewed before deposit. Otherwise retain the implementation as post-deposit demonstrator work and
preserve the current thesis wording that the application and external transfer remain future work.

A rights ambiguity, a need to retune on external results, an invalid comparison, or insufficient
time for review triggers NO-GO without weakening the completed SMB study.
