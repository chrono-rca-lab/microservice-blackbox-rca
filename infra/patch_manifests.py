#!/usr/bin/env python3
"""Patch Online Boutique kubernetes-manifests.yaml for fault injection.

Downloads the upstream manifest if not already present, then applies three
modifications to every Deployment so fault scripts can run inside the
service containers:

  1. Removes readOnlyRootFilesystem: true  (disk_hog needs writable paths)
  2. Adds NET_ADMIN capability             (net_delay needs tc / iptables)
  3. Adds an emptyDir volume at /tmp       (gives dd a writable block-backed path)

Usage:
    python infra/patch_manifests.py                          # uses defaults
    python infra/patch_manifests.py --input  infra/boutique-manifests.yaml \\
                                    --output infra/boutique-manifests-patched.yaml
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import yaml

MANIFESTS_URL = (
    "https://raw.githubusercontent.com/GoogleCloudPlatform/"
    "microservices-demo/main/release/kubernetes-manifests.yaml"
)

SCRIPT_DIR = Path(__file__).parent


def _patch_container(container: dict) -> None:
    """Mutate a single container spec in-place."""
    sc = container.setdefault("securityContext", {})

    # 1. Remove readOnlyRootFilesystem restriction
    sc.pop("readOnlyRootFilesystem", None)
    # Clean up the securityContext if it's now empty
    if not sc:
        container.pop("securityContext", None)

    # 2. Add NET_ADMIN capability
    sc = container.setdefault("securityContext", {})
    caps = sc.setdefault("capabilities", {})
    add_list = caps.setdefault("add", [])
    if "NET_ADMIN" not in add_list:
        add_list.append("NET_ADMIN")

    # 3. Add /tmp volumeMount (idempotent)
    mounts = container.setdefault("volumeMounts", [])
    if not any(m.get("mountPath") == "/tmp" for m in mounts):
        mounts.append({"name": "tmp-vol", "mountPath": "/tmp"})


def _patch_deployment(doc: dict) -> None:
    """Mutate a Deployment document in-place."""
    pod_spec = doc["spec"]["template"]["spec"]

    for c in pod_spec.get("containers", []):
        _patch_container(c)
    for c in pod_spec.get("initContainers", []):
        _patch_container(c)

    # Add tmp-vol emptyDir (idempotent)
    volumes = pod_spec.setdefault("volumes", [])
    if not any(v.get("name") == "tmp-vol" for v in volumes):
        volumes.append({"name": "tmp-vol", "emptyDir": {}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=str(SCRIPT_DIR / "boutique-manifests.yaml"),
        help="Path to upstream kubernetes-manifests.yaml (downloaded if missing)",
    )
    parser.add_argument(
        "--output", default=str(SCRIPT_DIR / "boutique-manifests-patched.yaml"),
        help="Path to write the patched manifest",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    # Download if missing
    if not input_path.exists():
        print(f"[patch] Downloading {MANIFESTS_URL} → {input_path}")
        urllib.request.urlretrieve(MANIFESTS_URL, input_path)
    else:
        print(f"[patch] Using existing {input_path}")

    docs = [d for d in yaml.safe_load_all(input_path.read_text()) if d is not None]

    patched_count = 0
    for doc in docs:
        if doc.get("kind") == "Deployment":
            _patch_deployment(doc)
            name = doc.get("metadata", {}).get("name", "?")
            print(f"[patch]   patched deployment/{name}")
            patched_count += 1

    output_path.write_text(
        yaml.dump_all(docs, default_flow_style=False, allow_unicode=True)
    )
    print(f"[patch] {patched_count} deployment(s) patched → {output_path}")


if __name__ == "__main__":
    main()
