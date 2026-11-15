"""The per-factor Reward-State analytical reference."""
from experiments.belief_quotient_persistent_nuisance_jax_2026_11.design import controller_references
import json

if __name__ == "__main__":
    print(json.dumps(controller_references(), indent=2))
