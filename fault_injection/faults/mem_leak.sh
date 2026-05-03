#!/bin/sh
# Ramp up heap use over DURATION, cap at MAX_MB. Python path if python3 exists;
# otherwise shell + base64 growth (slower). Tuned to grow the series without OOM.
DURATION="${DURATION:-60}"
MAX_MB="${MAX_MB:-150}"

echo "[mem_leak] allocating up to ${MAX_MB}MB at 2MB/s for ${DURATION}s"

if command -v python3 >/dev/null 2>&1; then
    python3 -c "
import time
leak = []
end = time.time() + $DURATION
max_bytes = $MAX_MB * 1024 * 1024
total = 0
    chunk = 2 * 1024 * 1024
while time.time() < end:
    if total + chunk <= max_bytes:
        leak.append(b'x' * chunk)
        total += chunk
    time.sleep(1)
"
else
    # no python: stuff random bytes into a string (wasteful but works)
    data=""
    end_time=$(( $(date +%s) + DURATION ))
    while [ "$(date +%s)" -lt "$end_time" ]; do
        chunk=$(dd if=/dev/urandom bs=1024 count=512 2>/dev/null | base64)
        data="${data}${chunk}"
        sleep 1
    done
fi

echo "[mem_leak] done"
