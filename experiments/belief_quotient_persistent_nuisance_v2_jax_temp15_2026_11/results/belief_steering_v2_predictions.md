# V2 numerical reference sensitivity

| Variant | γ | N | Bellman residual | Converged | Intact occupancy |
|---:|---:|---:|---:|---|---:|
| v1 | 0.97 | 60 | 9.84e-10 | True | 0.945007 |
| v1 | 0.97 | 100 | 9.88e-10 | True | 0.945007 |
| v1 | 0.99 | 60 | 9.9e-10 | True | 0.945007 |
| v1 | 0.99 | 100 | 9.92e-10 | True | 0.945007 |
| v1 | 0.995 | 60 | 9.99e-10 | True | 0.945007 |
| v1 | 0.995 | 100 | 9.99e-10 | True | 0.945007 |
| v2 | 0.97 | 60 | 9.95e-10 | True | 0.564026 |
| v2 | 0.97 | 100 | 9.82e-10 | True | 0.564026 |
| v2 | 0.99 | 60 | 9.98e-10 | True | 0.564026 |
| v2 | 0.99 | 100 | 9.99e-10 | True | 0.564026 |
| v2 | 0.995 | 60 | 9.98e-10 | True | 0.564026 |
| v2 | 0.995 | 100 | 9.99e-10 | True | 0.564026 |
| v3 | 0.97 | 60 | 9.72e-10 | True | 0.501617 |
| v3 | 0.97 | 100 | 9.86e-10 | True | 0.486389 |
| v3 | 0.99 | 60 | 9.98e-10 | True | 0.487213 |
| v3 | 0.99 | 100 | 9.96e-10 | True | 0.504120 |
| v3 | 0.995 | 60 | 9.99e-10 | True | 0.483795 |
| v3 | 0.995 | 100 | 1e-09 | True | 0.495300 |

Grid sensitivity (maximum occupancy difference across targets and λ, common random streams):

{
  "v1_g0.97": {
    "max_occupancy_grid_difference": 0.0,
    "bellman_converged": true,
    "grid_agreement_within_005": true
  },
  "v1_g0.99": {
    "max_occupancy_grid_difference": 0.0,
    "bellman_converged": true,
    "grid_agreement_within_005": true
  },
  "v1_g0.995": {
    "max_occupancy_grid_difference": 0.0,
    "bellman_converged": true,
    "grid_agreement_within_005": true
  },
  "v2_g0.97": {
    "max_occupancy_grid_difference": 0.0,
    "bellman_converged": true,
    "grid_agreement_within_005": true
  },
  "v2_g0.99": {
    "max_occupancy_grid_difference": 0.0,
    "bellman_converged": true,
    "grid_agreement_within_005": true
  },
  "v2_g0.995": {
    "max_occupancy_grid_difference": 0.0,
    "bellman_converged": true,
    "grid_agreement_within_005": true
  },
  "v3_g0.97": {
    "max_occupancy_grid_difference": 0.31768798828125,
    "bellman_converged": true,
    "grid_agreement_within_005": false
  },
  "v3_g0.99": {
    "max_occupancy_grid_difference": 0.332122802734375,
    "bellman_converged": true,
    "grid_agreement_within_005": false
  },
  "v3_g0.995": {
    "max_occupancy_grid_difference": 0.309356689453125,
    "bellman_converged": true,
    "grid_agreement_within_005": false
  }
}

Full curves are in the companion JSON and apply to both epsilon studies. Bellman convergence alone does not demonstrate grid convergence.
