#!/bin/sh
# Stress CPU with a single busy-loop (POSIX sh, no external tools needed).
# Runs inside the target container so cAdvisor attributes usage to it.
# One worker is sufficient: it saturates the container's CPU limit, driving
# cpu_throttle_ratio to ~50-80% — a strong RCA signal — while leaving enough
# headroom for the service's liveness probe to respond.
DURATION="${DURATION:-60}"
N=1
echo "[cpu_hog] spawning ${N} busy-loop(s) for ${DURATION}s"

i=0
while [ "$i" -lt "$N" ]; do
    (while true; do :; done) &
    i=$((i + 1))
done

sleep "$DURATION"
kill $(jobs -p) 2>/dev/null || true
echo "[cpu_hog] done"
