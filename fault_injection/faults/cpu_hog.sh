#!/bin/sh
# Stress CPU with a single busy-loop (POSIX sh, no external tools needed).
# Runs inside the target container so cAdvisor attributes usage to it.
# One busy loop fills the cgroup quota: throttle ratio jumps but the probe usually
# still gets enough CPU.
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
