"""Read-only imports from the pinned paper pulse driver."""
import importlib
import os
from pathlib import Path
import subprocess

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

REVISION = "45517b13a67d746048a002f9db52ea55a654b219"
STUDY = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("PULSE_REFERENCE", STUDY / "artifacts" / "arch_pulse_reference"))
PACKAGE = "experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11"
if subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip() != REVISION:
    raise ValueError("pulse reference checkout revision differs")
package = importlib.import_module(PACKAGE)
package.__path__.insert(0, str(SOURCE / PACKAGE.replace(".", "/")))
core = importlib.import_module(PACKAGE + ".steering_core")
pulse = importlib.import_module(PACKAGE + ".steering_pulse")
