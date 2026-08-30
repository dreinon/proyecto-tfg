# Experiment configurations

Store versioned, human-readable configurations here. A configuration should identify the dataset
and split manifest, degradation pipeline, model/weights, seed, training or inference parameters,
evaluation protocol, and output run identifier.

The final SMB rerun is frozen by `degradations/staff-scale-score-v2.yaml`,
`experiments/smb-pretrained-evaluation-v2.yaml`, and the evidence-backed gate under
`smb-evaluation-v2/`. The v1 counterparts remain immutable development-pilot evidence and must not
be overwritten or silently relabelled as final.
