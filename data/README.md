# Data management

This is the single data namespace for the project. It contains only small, reproducibility-critical
files:

- `sources/`: tracked descriptors for external datasets, including immutable revisions, access,
  licence, citation, official splits, and observed schema;
- `manifests/`: tracked selections or project-defined splits, with one row per independent source
  item and enough information to reproduce the selection.

The image datasets themselves do not belong in Git. Hugging Face data should be loaded through
`datasets.load_dataset` and left in the normal Hugging Face cache (or an explicitly recorded cache
location on a remote runtime). Do not copy the SMB images into a second repository-local dataset
tree.

Every project-defined manifest must record its source descriptor and revision, creation command,
seed where relevant, exclusions, and grouping key. Split independent source scores or documents
before extracting pages or patches. Do not reinterpret an official benchmark split as training
data without recording and justifying that protocol change.

Generated contents of `raw/`, `interim/`, and `processed/` are ignored by Git if a pipeline later
needs temporary local materialization.
