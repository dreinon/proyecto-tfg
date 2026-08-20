# Data manifests

Place small, versioned dataset selections and project-defined splits here. Prefer a transparent
CSV, JSONL, or Parquet file plus a short YAML descriptor containing:

- source key and immutable revision;
- creation command and code revision;
- grouping unit (score, work, document, page, or patch);
- upstream and project-defined split names;
- deterministic seed, if applicable;
- exclusions and their reasons;
- row count and checksum.

SMB already supplies an official `test` split. A manifest may freeze exclusions or evaluation
subgroups, but must not silently relabel benchmark examples as training or validation data.
