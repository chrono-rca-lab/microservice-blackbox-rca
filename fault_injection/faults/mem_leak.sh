#!/bin/sh
# Allocate memory continuously for DURATION seconds.
# Uses python3 if available (Python services); falls back to shell + base64 accumulation.
DURATION="${DURATION:-60}"

echo "[mem_leak] allocating memory for ${DURATION}s"

if command -v python3 >/dev/null 2>&1; then
    python3 -c "
import time
leak = []
end = time.time() + $DURATION
while time.time() < end:
    leak.append(b'x' * 10 * 1024 * 1024)
    time.sleep(1)
"
else
    # Shell fallback: accumulate base64-encoded random data into a variable
    data=""
    end_time=$(( $(date +%s) + DURATION ))
    while [ "$(date +%s)" -lt "$end_time" ]; do
        chunk=$(dd if=/dev/urandom bs=1024 count=512 2>/dev/null | base64)
        data="${data}${chunk}"
        sleep 1
    done
fi

echo "[mem_leak] done"
