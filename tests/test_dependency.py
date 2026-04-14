"""Tests for rca_engine.dependency."""

from rca_engine.dependency import get_dependency_graph, has_path, ONLINE_BOUTIQUE_DEPENDENCIES


def test_get_dependency_graph_returns_copy():
    g1 = get_dependency_graph()
    g2 = get_dependency_graph()
    g1["frontend"].append("FAKE")
    assert "FAKE" not in g2["frontend"]


def test_has_path_direct():
    g = get_dependency_graph()
    assert has_path(g, "frontend", "adservice")
    assert has_path(g, "cartservice", "redis-cart")


def test_has_path_transitive():
    g = get_dependency_graph()
    # frontend -> checkoutservice -> paymentservice
    assert has_path(g, "frontend", "paymentservice")
    # frontend -> cartservice -> redis-cart
    assert has_path(g, "frontend", "redis-cart")
    # frontend -> recommendationservice -> productcatalogservice
    assert has_path(g, "frontend", "productcatalogservice")


def test_has_path_no_reverse():
    g = get_dependency_graph()
    assert not has_path(g, "adservice", "frontend")
    assert not has_path(g, "redis-cart", "cartservice")
    assert not has_path(g, "paymentservice", "checkoutservice")


def test_has_path_self():
    g = get_dependency_graph()
    assert has_path(g, "frontend", "frontend")


def test_has_path_unknown_node():
    g = get_dependency_graph()
    assert not has_path(g, "frontend", "nonexistent")
    assert not has_path(g, "nonexistent", "frontend")


def test_no_path_between_leaf_services():
    g = get_dependency_graph()
    assert not has_path(g, "adservice", "paymentservice")
    assert not has_path(g, "emailservice", "currencyservice")


def test_all_11_services_present():
    """Dependency graph should include all 11 Online Boutique services."""
    expected = {
        "frontend", "cartservice", "productcatalogservice", "currencyservice",
        "paymentservice", "shippingservice", "emailservice", "checkoutservice",
        "recommendationservice", "adservice", "redis-cart",
    }
    assert set(ONLINE_BOUTIQUE_DEPENDENCIES.keys()) == expected


def test_leaf_services_have_no_dependencies():
    """Leaf services should have empty dependency lists."""
    leaves = [
        "productcatalogservice", "currencyservice", "paymentservice",
        "shippingservice", "emailservice", "adservice", "redis-cart",
    ]
    g = get_dependency_graph()
    for svc in leaves:
        assert g[svc] == [], f"{svc} should be a leaf"


def test_frontend_is_root():
    """Frontend should have the most dependencies (it's the entry point)."""
    g = get_dependency_graph()
    assert len(g["frontend"]) >= 7


def test_checkoutservice_calls_payment():
    """Checkout calls payment -- critical for propagation tests."""
    g = get_dependency_graph()
    assert "paymentservice" in g["checkoutservice"]


def test_has_path_empty_graph():
    """Empty graph should always return False (except self)."""
    assert has_path({}, "a", "b") is False
    assert has_path({}, "a", "a") is True


def test_has_path_with_custom_cyclic_graph():
    """Test BFS with a custom cyclic graph."""
    g = {"a": ["b"], "b": ["c"], "c": ["a"]}  # cycle
    assert has_path(g, "a", "c") is True
    assert has_path(g, "c", "b") is True  # c -> a -> b
