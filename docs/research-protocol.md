# Super-resolution research protocol

This is the working experimental contract. GSD phase plans may refine it, but a change affecting
comparability must be recorded before interpreting results.

## Research objective

Determine whether selected super-resolution techniques can recover useful visual detail in
degraded digitized music scores without introducing musically meaningful errors. If the
post-baseline decision gate justifies domain-specific fine-tuning, additionally determine whether
that bounded adaptation improves the balance over simple and pretrained baselines.

The core contribution is a controlled, reproducible comparison in the music-score domain—not the
invention of a new architecture. PDF tooling, a web application, and OMR are optional extensions.
The contribution must be expressed through the domain-specific protocol, notation-failure analysis,
and evidence-backed professional decision framework rather than a model leaderboard alone.

## Dataset protocol

SMB (`PRAIG/SMB`) is the primary evaluation source. Its immutable Hugging Face revision and
upstream metadata are recorded in `data/sources/smb.yaml`; the descriptor reports manual gating,
CC BY-NC 4.0, and 685 examples in the sole official split `test`, pending authenticated audit.
Preserve that reported role as a benchmark: do not tune models, select checkpoints, or fit
degradation parameters on SMB. A separate licensed, group-disjoint training/validation source is
required for domain fine-tuning. The v1 protocol does not permit repurposing SMB's sole official
`test` split for training, validation, checkpoint selection, or any other adaptive decision.

For every dataset, record:

- canonical name, owner, URL/identifier, version/date, licence, and citation;
- acquisition command and cryptographic checksums where practical;
- file/page counts, formats, resolutions, colour modes, corruption, duplicates, and exclusions;
- meaningful domains such as engraving/handwritten source, publisher/style, typography, staff
  size, notation density, language/text, scan vs. born-digital origin, and quality;
- exact split manifest.

When project-defined splits are needed, split independent source scores/works/documents first.
Only then derive pages, crops, or patches. Preserve official benchmark splits and document any
deliberate departure from them.
Grouping must prevent near-duplicate material and pages from the same score from crossing splits.
Keep test data untouched until the analysis protocol is fixed.

## Degradation protocol

Separate controlled synthetic degradation from evaluation on realistic low-quality scans. For
each pipeline, specify and version:

- enlargement factor;
- resampling kernel;
- blur kernel/type and parameter range;
- noise distribution and range;
- compression codec/quality;
- colour/bit-depth conversion;
- order of operations;
- border/cropping behaviour;
- sampled parameters and seed for every generated example.

Include unit and visual checks for dimensions, value range, determinism, alignment with ground
truth, and absence of accidental data leakage.

## Comparison ladder

Proceed only as far as evidence and time permit:

1. Nearest, bilinear, and bicubic interpolation.
2. A small, justified set of current pretrained SR models representing meaningfully different
   approaches.
3. Domain fine-tuning of only the most informative candidates.
4. Targeted ablations needed to explain an observed effect.
5. At most one bounded applicability enhancement after the image-domain conclusions are stable:
   preferably a provenance-safe realistic-scan stress test or a predeclared blinded specialist
   pilot; OMR only if it has greater justified value and a valid protocol.

Avoid collecting many models with shallow analysis. Selection criteria, compute budget, licence,
input assumptions, and expected scientific value must be explicit.

## Evaluation protocol

Use identical aligned test inputs and aggregation for every comparable method. Report results
overall and by relevant subgroup/degradation severity.

Quantitative evaluation should cover complementary properties rather than rely on one score:

- pixel fidelity such as PSNR;
- structural similarity such as SSIM;
- a justified perceptual measure if compatible with the evaluation question;
- runtime, peak memory, parameter count, and/or compute where informative;
- domain-specific measurements or OMR only when their validity is established.

Qualitative evaluation must use a fixed sampling rule and the predeclared failure taxonomy:

- broken, removed, thickened, or hallucinated staff lines;
- missing/deformed noteheads, stems, beams, flags, rests, clefs, accidentals, articulations,
  dynamics, slurs, ties, barlines, or ledger lines;
- unintended joins or separations;
- altered text, lyrics, fingering, rehearsal marks, or digits;
- plausible-looking but musically different content.

Keep both positive and negative examples. If human ratings are used, define the rubric, sampling,
blinding/order, raters, disagreements, and aggregation in advance.

## Run record

Every reported run must be recoverable from:

- run identifier and timestamp;
- Git revision and dirty-state flag;
- dataset/split/degradation manifest identifiers;
- model source, version, weights/checkpoint checksum, and licence;
- complete configuration and random seeds;
- Python/dependency/hardware/runtime snapshot;
- training/inference logs and selection criterion;
- raw per-item metrics plus aggregate tables;
- output artefact paths and any exclusion/failure notes.

Use TensorBoard initially. Add a heavier experiment service only if local logs and structured run
manifests become inadequate.

## Local and Kaggle environments

Local WSL development is CPU-first and uses Python 3.12 through `uv`. It is intended for data
inspection, small samples, unit tests, metrics, pipeline validation, and lightweight inference.

Kaggle is the planned accelerator environment. For each Kaggle experiment:

1. record notebook/kernel ID and revision;
2. record accelerator model, Python, PyTorch, CUDA, and installed-package snapshot;
3. install only missing project dependencies rather than replacing the runtime blindly;
4. capture the project Git revision and transfer a fixed configuration/data manifest;
5. seed all relevant libraries and record determinism limitations;
6. export logs, configurations, metrics, and selected checkpoints before the session ends;
7. verify locally that exported results match the declared run.

Never place Kaggle API credentials in notebooks, repositories, logs, or artefacts.
Provide gated Hugging Face access through a Kaggle Secret (for example `HF_TOKEN`), never as a
literal token in a notebook. Use the same immutable dataset revision locally and remotely.

## Decision gates

- Do not benchmark models until dataset provenance and leakage-safe splits are validated.
- Do not fine-tune until degradation and interpolation/pretrained baselines are reproducible.
- Do not inspect the final test set to choose models or hyperparameters.
- Do not start optional OMR/application work until the core SR evaluation answers the approved
  objectives.
- After the controlled core is reconciled, record a tutor-approved GO/NO-GO decision on one bounded
  applicability enhancement. A NO-GO is valid when evidence, rights, time, or methodological
  quality are insufficient.
- Do not write a strong conclusion from aggregate metrics without inspecting domain-specific
  failures.
