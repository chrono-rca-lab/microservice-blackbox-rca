"""Pin a Boutique deployment onto the experiment-target node before injecting a fault.

`--isolate` in the runners does this so cAdvisor anomalies on the victim pod are
easier to read (no contention from unrelated services on the same machine).

Needs a node with label `role=experiment-target` (Machine 4 in our VCL setup);
see infra/VCL_README.md.

Typical pattern:

    sel = move_to_experiment_node("emailservice")
    try:
        ...
    finally:
        restore_service("emailservice", sel)
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
    """Pin the workload to experiment-target and wait until the rollout settles.

    Call this before baseline so “normal” traffic is scraped on the new node.
    Returned dict is the previous nodeSelector; pass it back to restore_service().
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
    """Put the deployment back where it came from."""
    print(f"  [isolate] restoring {service} → {original_selector} …")
    _patch_node_selector(service, original_selector)
    _wait_rollout(service)
    print(f"  [isolate] {service} restored")
