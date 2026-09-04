"""Run deterministic A1/A2 and E3 structural audits without training."""

from harness.artifacts import RunArtifacts
from harness.context import RunContext

from experiments.mess3_factored_cycle_1.reference import structural_audit_report


def run(context: RunContext):
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    report = structural_audit_report()
    outputs.write_json("audit_status.json", report)
    if report["status"] != "passed":
        raise RuntimeError("factored MESS3 structural audits failed")
    return report
