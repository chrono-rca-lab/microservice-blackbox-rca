#!/bin/sh
# Saturate all CPUs with shell busy-loops (POSIX sh, no external tools needed).
# Runs inside the target container so cAdvisor attributes usage to it.
DURATION="${DURATION:-60}"

N=$(nproc 2>/dev/null || grep -c processor /proc/cpuinfo 2>/dev/null || echo 1)
echo "[cpu_hog] spawning ${N} busy-loop(s) for ${DURATION}s"

i=0
while [ "$i" -lt "$N" ]; do
    (while true; do :; done) &
    i=$((i + 1))
done

sleep "$DURATION"
kill $(jobs -p) 2>/dev/null || true
echo "[cpu_hog] done"
