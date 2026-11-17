"""Reproduce minimal pinned protocol extracts without importing launch/upload CLIs."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LADDER = '45517b13a67d746048a002f9db52ea55a654b219'
MOVING = '3504bfb72455a411b725ef368a18ff892760555f'
PREFIX = 'experiments/belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11/'


def generate():
    provenance = {}

    def extract(revision, path, names=None):
        source = subprocess.check_output(['git', '-C', str(REPO), 'show', revision+':'+path], text=True)
        provenance[path] = dict(revision=revision, sha256=hashlib.sha256(source.encode()).hexdigest())
        lines = source.splitlines(True)
        segments = []
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (names is None or node.name in names):
                start = min([node.lineno]+[d.lineno for d in getattr(node, 'decorator_list', [])])-1
                segments.append(''.join(lines[start:node.end_lineno]))
        return '\n\n'.join(segments)+'\n'

    common = 'from functools import partial\nimport math\nimport jax\nimport jax.numpy as jnp\nimport numpy as np\nfrom pathlib import Path\n'
    trainer = 'from experiments.belief_quotient_history_jax_2026_11 import models, env\n'
    environment = 'from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import environment as v2_environment\n'
    outputs = {}
    outputs['steer_core.py'] = common+trainer+"MODES = ('point', 'mirror', 'rho_flip', 'rho_point')\nLAMBDAS = (0., .25, .4, .45, .5, .55, .6, .75, .9, 1.)\nAFFINE_RCOND = 1e-6\n"+extract(LADDER, PREFIX+'steering_core.py')
    outputs['steer_prediction.py'] = common+environment+extract(LADDER, PREFIX+'steering_prediction.py', {'target', 'Policy'})
    outputs['steer_pulse.py'] = common+trainer+environment+'from .steer_core import instrumented_step, pulse_branch\nfrom .steer_prediction import target\n'+extract(LADDER, PREFIX+'steering_pulse.py')
    outputs['steer_storage.py'] = 'import numpy as np\n'+extract(LADDER, PREFIX+'steering_storage.py', {'headline', 'metrics_summary', 'pulse_summary', 'compact'})
    driver = extract(LADDER, PREFIX+'belief_steering.py', {'run'})
    numerical = driver[driver.index('    rows, length ='):driver.index('    harness_root =')]
    report = driver[driver.index('    report = dict('):driver.index('    path.write_text')]
    report = report.replace('study=study, ', '').replace(' checkpoint=checkpoint,', '')
    a, b = report.index('        provenance='), report.index('        protocol=')
    report = report[:a]+report[b:]
    outputs['steer_ladder.py'] = common+'import time\nfrom .steer_core import AFFINE_RCOND, LAMBDAS, MODES, empty_maps, fit_maps, metrics, probe_scores, rollout\nfrom .steer_prediction import Policy, target\nfrom .steer_pulse import run_pulse\n\ndef run(spec, process, params, raw, variant, seed, smoke, environment):\n    start = time.perf_counter()\n'+numerical+report+'    return report\n'
    base_path = 'experiments/two_factor_moved_reward_steering_v2_jax_2026_11/steering.py'
    base_names = {'factor_beliefs', 'forward', 'decoded_factor_beliefs', '_keys', 'collect', 'collect_rows', 'affine_fit', 'r_squared', 'map_targets', 'learned_action_map', 'anchor_states', 'null_direction'}
    constants = "HORIZON = 1024\nWARMUP = 64\nFIT_ROWS, TEST_ROWS = 60000, 80000\nSTRENGTHS = (0., .25, .5, .75, .8, .85, .9, .95, 1., 1.25, 1.5, 2., 3.)\nTARGETS = ('marginal', 'joint')\nACTION_MAPS = ('myopic', 'discounted')\nGAMMA = .99\nACTION_TABLE = np.asarray(ACTION_PAIRS, dtype=np.int32)\n"
    outputs['steer_moving_base.py'] = common+trainer+'from experiments.belief_quotient_history_jax_2026_11 import env as hmm\nfrom experiments.reward_factors_history_jax_2026_11.process import ACTION_PAIRS\n'+constants+extract(MOVING, base_path, base_names)
    coupled = 'experiments/two_factor_coupled_moving_reward_steering_v2_jax_2026_11/steering.py'
    names = {'source_process', 'joint_transitions', 'reward_components', 'reward_schedule', 'q_tables', 'decoded_joint', 'belief_target', 'steering_shift', '_evaluate', 'evaluate', 'analyze'}
    imports = common+'import json\nimport hashlib\n'+trainer+'from experiments.belief_quotient_history_jax_2026_11 import env as hmm\nfrom experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment as eps02\nfrom experiments.reward_factors_persistent_nuisance_jax_2026_11.shared import factor_process\nfrom . import steer_moving_base as base\n'
    constants = "REWARDS = ('moving',)\nSTEERED_FACTORS = {'moving': (False, True)}\nPERIOD = 256\nHORIZON = base.HORIZON\nGAMMA = base.GAMMA\nACTION_MAPS = base.ACTION_MAPS\nTARGETS = base.TARGETS\nSTRENGTHS = base.STRENGTHS\n_FIRST, _SECOND = np.divmod(np.arange(9), 3)\nSOURCE_STUDIES = {f'reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_{algorithm}_2026_11': (eps02, algorithm) for algorithm in ('ppo', 'sac_autoalpha')}\n"
    outputs['steer_moving.py'] = imports+constants+extract(MOVING, coupled, names)
    outputs['steer_reference.py'] = common+extract(LADDER, 'experiments/belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11/steering_reference.py', {'simulate'})
    for name, body in outputs.items():
        (ROOT/name).write_text(('"""Pinned protocol extract; generated by vendor_steering.py. See steering_sources.json."""\n\n'+body).rstrip()+'\n')
    provenance['adaptations'] = ['import relocation and minimal function extraction', 'ladder driver excludes restoration/publication/provenance IO only', 'moving excludes XOR via REWARDS; replaces source study inventory only', 'moving base load_params supplied by strict verified adapter']
    (ROOT/'steering_sources.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'Generated {len(outputs)} minimal protocol modules')


if __name__ == '__main__':
    generate()
