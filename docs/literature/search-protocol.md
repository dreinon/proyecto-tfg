# Focused super-resolution evidence review protocol

## Scope and review questions

This protocol governs the Phase 1 evidence base for a focused, layered review of super-resolution
(SR) for digitized music scores. It asks which concepts and evidence are needed to design a later
controlled comparison without choosing a learned method or checkpoint now:

1. Which classical, CNN, transformer, perceptual, blind, and generative approaches establish the
   relevant SR design space?
2. Which degradation assumptions separate controlled reconstruction from real-world restoration?
3. What direct document and music-score evidence exists, and what does it not establish?
4. Which evaluation evidence warns against equating visual realism or aggregate fidelity with
   correct musical notation?

The search covers landmark evidence and current primary work through 2026-08-18. It is deliberately
focused and not exhaustive. The matrix supports protocol design; it is not a ranking, a benchmark
result, or a model-selection record.

## Sources and exact queries

Searches were run against the following primary or official surfaces. Web search was used only to
discover canonical records; factual rows resolve to the DOI, publisher proceedings page, official
dataset card, or official repository linked by the paper.

| Surface | Exact query strings |
|---|---|
| IEEE Xplore / DOI | `"single image super-resolution" SRCNN`; `"Cubic convolution interpolation for digital image processing"`; DOI lookups `10.1109/TASSP.1981.1163711`, `10.1109/TPAMI.2015.2439281`, and `10.1109/ICCVW54120.2021.00210` |
| CVF Open Access | `site:openaccess.thecvf.com super-resolution EDSR`; `SRGAN`; `ESRGAN`; `perception distortion tradeoff`; `SwinIR`; `HAT image super-resolution`; `Real-ESRGAN`; `DocRes`; `CVPR 2026 one-step diffusion super-resolution`; `CVPR 2026 document restoration`; `CVPR 2026 text image super-resolution` |
| DOI / publisher search | `"Super-resolution in Music Score Images by Instance Normalization"`; `"Task-Driven Real-World Super-Resolution of Document Scans"` |
| ISMIR / Zenodo / Hugging Face | `"Sheet Music Benchmark" OMR-NED`; `site:huggingface.co/datasets/PRAIG/SMB`; DOI lookup `10.5281/zenodo.17811446` |
| ACM DL / SpringerLink | `document image super-resolution`; `music score image super-resolution`; retained only when a canonical primary record added a missing layer |
| Official code repositories | Exact paper title or repository URL linked from the canonical paper page; licence and checkpoint statements were recorded only when visible in official material |

No general web-search snippet is used as evidence. ArXiv is retained only as a duplicate or when no
archival version exists; the included rows in this initial matrix all resolve to archival or
official records.

## Search dates

- Main layered discovery and primary-record verification: 2026-08-17 and 2026-08-18.
- Matrix verification date: 2026-08-18 for every included row.
- Temporal boundary: primary evidence available through 2026-08-18, including CVPR 2026 records.
- Later changes to code, checkpoints, licences, or compatibility do not silently update this file;
  they require a dated new screening/matrix revision.

## Citation chasing

Backward citation chasing started from the direct score-SR paper, the SMB/OMR paper, SRCNN, EDSR,
SwinIR, Real-ESRGAN, DocRes, and the 2025 real-document SR paper. Forward chasing used their
publisher “cited by/related material” surfaces and the 2026 CVF proceedings. A candidate was added
only if it supplied a missing layer, a primary contradiction/limitation, or newer evidence that
could change the framing. Surveys were used to discover terminology and primary papers, then logged
as `discovery_only`; no claim candidate cites a survey.

## Deduplication

The canonical identity is, in order: normalized DOI; canonical archival proceedings URL when no DOI
was found; official dataset/repository URL for documentation. DOI strings are lower-cased and URL
prefixes removed. Without a DOI, titles are Unicode-normalized, lower-cased, stripped of punctuation,
and collapsed on whitespace, then checked against author and year. Conference/preprint and journal
versions are linked with `duplicate_of`; the fuller archival version is retained. Duplicate and
excluded records are never deleted from `screening-log.csv`.

## Inclusion criteria

- Original research paper or official dataset/model/code documentation with stable identity.
- Supplies at least one predeclared layer: classical interpolation; fidelity/perceptual SR;
  CNN/transformer foundations; direct document/score evidence; controlled/real degradation;
  evaluation; hallucination or semantic-structure risk; or current 2026 context.
- Reports enough method, data/degradation, evaluation, and limitation information to populate every
  D-08 field without guessing. `not_reported` and `not_applicable` are evidence states, not blanks.
- Directly informs this TFG's later controlled, score-specific evaluation or its claim boundaries.

## Exclusion criteria

- Survey, tutorial, aggregator, blog, search snippet, or AI-generated summary offered as factual
  support. Surveys may remain `discovery_only`.
- Duplicate publication where a canonical fuller version is already retained.
- Unrelated restoration task, non-image SR, or architecture catalogue that adds no missing layer.
- Source that cannot be resolved to a DOI or canonical official URL.
- Model/checkpoint marketing without sufficient provenance, or work whose only value would be to
  pre-empt the Phase 3 selection decision.
- SMB measurement, audit outcome, SR output, or metric: all such outcome evidence is prohibited in
  this phase.

## Coverage targets

The matrix must contain primary/official support for all of these tags:
`classical_interpolation`, `fidelity_sr`, `perceptual_sr`, `cnn_foundation`,
`transformer_foundation`, `document_or_score`, `controlled_degradation`, `real_degradation`,
`evaluation`, `semantic_risk`, and `current_2026`. At least one 1981-or-earlier landmark and one
2026 paper are required. Direct score-SR and official SMB context must be present, while absence of
additional direct work is described only as a search limitation, never as proof of nonexistence.

## Saturation and stop rule

Phase 1 stops when every coverage tag has at least one primary/official row, the direct score-SR
seed and SMB paper/card have been backward- and forward-chased, and one further query/citation-chase
pass adds no new decision-relevant layer. The final pass on 2026-08-18 produced only duplicates,
discovery-only surveys, or method variants that did not close a coverage gap. This is focused
saturation, not a systematic-review claim and not evidence that no other relevant paper exists.

## Evidence and claim handling

`sota-matrix.csv` records verified metadata separately from `relevance_to_tfg`, which is an explicit
project interpretation. `reported_results` is limited to what the primary source reports; it does
not imply replication. `claim-candidates.csv` contains bounded, pending candidates using the shared
claim-evidence shape. Pending rows are not thesis-ready: review, wording, and critical comparison
remain outstanding. CSV cells are quoted by the CSV format and may not start with spreadsheet
formula characters.

## Known limitations

- Database searching was focused and English-query-led; it may miss non-English titles, papers with
  different terminology, and unindexed venues.
- The 2019 direct score-SR paper's accessible metadata is much less detailed than later open papers;
  missing details are recorded as `not_reported`, not inferred.
- The official ISMIR 2025 index lists the SMB paper at pages 604-611 while the official dataset-card
  citation lists 618-625. The matrix retains the conference index pagination and records the
  conflict instead of silently choosing both.
- Repository and licence status can change. Paper access terms, code licence, weight identity, and
  executable compatibility are distinct fields and must not be conflated.
- Natural-image, text, and document findings motivate checks but do not demonstrate behavior on
  music scores. SMB remains reported context pending the separate authenticated audit.

## Phase 3 refresh

Before any learned-method decision, Phase 3 must re-run targeted searches from the 2026 boundary,
verify official repositories, exact weight/checkpoint identities and checksums, licences, input and
tiling assumptions, maintenance state, and Kaggle compatibility. It must record its own screening
decision rather than treating inclusion here as candidacy or selection.

## Phase 5 synthesis

Phase 5 must critically synthesize only reviewed claim-evidence rows into the thesis, reconcile the
refresh and actual experimental evidence, cite every promoted claim, retain negative findings, and
separate source-reported results from this TFG's results and interpretations.
