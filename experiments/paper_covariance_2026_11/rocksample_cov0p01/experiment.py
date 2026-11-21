from ..training import run_rocksample


def run(context):
    return run_rocksample(context, weight=0.01)
