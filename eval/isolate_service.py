"""Move a boutique service to the dedicated experiment-target node before injection.

Used via --isolate in run_experiment.py / run_experiment_slo.py to physically
isolate the service under test on its own machine. When emailservice is alone on
Machine 4, any cpu/mem anomaly detected by cAdvisor is purely from the injected
fault — not from currencyservice or paymentservice sharing the same node.

Requires a cluster node labelled  role=experiment-target  (Machine 4 in the VCL
4-node layout). See infra/VCL_README.md for setup instructions.

Usage from experiment runners:
    original = move_to_experiment_node("emailservice")
    try:
        ... run experiment ...
    finally:
        restore_service("emailservice", original)
"""

import json
import subprocess
from typing import Any


NAMESPACE = "boutique"
EXPERIMENT_ROLE = "experiment-target"


def _kubectl(*args: str) -> str:
    result = subprocess.run(
        ["kubectl", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _patch_node_selector(service: str, selector: dict[str, Any]) -> None:
    patch = json.dumps({"spec": {"template": {"spec": {"nodeSelector": selector}}}})
    _kubectl("patch", "deployment", service, "-n", NAMESPACE, "--type=merge", "-p", patch)


def _wait_rollout(service: str, timeout: int = 120) -> None:
    _kubectl(
        "rollout", "status", "deployment", service,
        "-n", NAMESPACE,
        f"--timeout={timeout}s",
    )


def move_to_experiment_node(service: str) -> dict[str, Any]:
    """Patch service deployment to run on the experiment-target node.

    Waits for the rollout to complete (old pod gone, new pod Ready) before
    returning. The experiment baseline should start only after this returns
    so the new node's metrics are captured in the normal distribution.

    Returns the original nodeSelector so restore_service can put it back.
    """
    raw = _kubectl(
        "get", "deployment", service,
        "-n", NAMESPACE,
        "-o", "jsonpath={.spec.template.spec.nodeSelector}",
    )
    original: dict[str, Any] = json.loads(raw) if raw.strip() else {}

    print(f"  [isolate] moving {service} → role={EXPERIMENT_ROLE} …")
    _patch_node_selector(service, {"role": EXPERIMENT_ROLE})
    _wait_rollout(service)
    print(f"  [isolate] {service} ready on experiment-target node")
    return original


def restore_service(service: str, original_selector: dict[str, Any]) -> None:
    """Restore service to its original node after the experiment."""
    print(f"  [isolate] restoring {service} → {original_selector} …")
    _patch_node_selector(service, original_selector)
    _wait_rollout(service)
    print(f"  [isolate] {service} restored")
