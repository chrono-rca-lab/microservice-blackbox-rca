"""Static service dependency graph for Online Boutique."""

from collections import deque

ONLINE_BOUTIQUE_DEPENDENCIES: dict[str, list[str]] = {
    "frontend": [
        "productcatalogservice",
        "currencyservice",
        "cartservice",
        "recommendationservice",
        "shippingservice",
        "checkoutservice",
        "adservice",
    ],
    "checkoutservice": [
        "productcatalogservice",
        "shippingservice",
        "paymentservice",
        "emailservice",
        "currencyservice",
        "cartservice",
    ],
    "recommendationservice": ["productcatalogservice"],
    "cartservice": ["redis-cart"],
    "productcatalogservice": [],
    "currencyservice": [],
    "paymentservice": [],
    "shippingservice": [],
    "emailservice": [],
    "adservice": [],
    "redis-cart": [],
}


def get_dependency_graph() -> dict[str, list[str]]:
    """Return a copy of the static dependency graph."""
    return {k: list(v) for k, v in ONLINE_BOUTIQUE_DEPENDENCIES.items()}


def has_path(graph: dict[str, list[str]], src: str, dst: str) -> bool:
    """Return True if there is a directed path from *src* to *dst* (BFS)."""
    if src == dst:
        return True
    visited: set[str] = set()
    queue: deque[str] = deque([src])
    while queue:
        node = queue.popleft()
        if node == dst:
            return True
        if node in visited:
            continue
        visited.add(node)
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                queue.append(neighbour)
    return False
