#!/bin/sh
# Inject network degradation via tc netem (preferred) or 30% packet loss via iptables.
# Needs NET_ADMIN; default Boutique pods don't have it, so this path mostly fails
# unless you patch capabilities. Batch runs usually skip it in favor of Chaos Mesh.
DURATION="${DURATION:-60}"
IFACE="${IFACE:-eth0}"

if command -v tc >/dev/null 2>&1; then
    echo "[net_delay] adding 200ms ± 50ms delay on ${IFACE} for ${DURATION}s (tc netem)"
    tc qdisc add dev "$IFACE" root netem delay 200ms 50ms
    sleep "$DURATION"
    tc qdisc del dev "$IFACE" root 2>/dev/null || true
elif command -v iptables >/dev/null 2>&1; then
    echo "[net_delay] adding 30% packet loss via iptables for ${DURATION}s"
    iptables -A OUTPUT -m statistic --mode random --probability 0.3 -j DROP
    sleep "$DURATION"
    iptables -D OUTPUT -m statistic --mode random --probability 0.3 -j DROP 2>/dev/null || true
else
    echo "[net_delay] ERROR: neither tc nor iptables available in this container" >&2
    echo "[net_delay] This fault requires NET_ADMIN capability. Use a privileged pod or skip." >&2
    exit 1
fi

echo "[net_delay] done"
