# Write-ups

This top-level directory contains manuscript-oriented material assembled from
one or more experiment leaves. Each subdirectory is a self-contained write-up
with:

- LaTeX source suitable for Overleaf;
- compact, machine-readable data snapshots used by its figures;
- scripts that regenerate derived charts; and
- provenance notes linking snapshots back to experiment and harness commits.

Large checkpoints and raw rollout artifacts remain under experiment
`artifacts/` directories or remote artifact storage. Experiment `results/`
remain the scientific source of truth; copied data here should be the smallest
auditable subset needed to reproduce the manuscript figures.
