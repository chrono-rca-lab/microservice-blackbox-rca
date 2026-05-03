#!/bin/sh
# Hammer disk I/O with dd.
# NOTE: Requires a writable path on a block-backed filesystem.
# readOnlyRootFilesystem containers often only have tmpfs; no real block writes.
# Use Chaos Mesh IOChaos or a writable mount if you need this to show up.
DURATION="${DURATION:-60}"

# Try to find a writable block-backed path
TMPFILE=""
for dir in /var/tmp /tmp /run /home; do
    if touch "$dir/.disk_hog_probe" 2>/dev/null; then
        rm -f "$dir/.disk_hog_probe"
        TMPFILE="$dir/disk_hog_$$.dat"
        break
    fi
done

if [ -z "$TMPFILE" ]; then
    echo "[disk_hog] ERROR: no writable block-backed path found (container has readOnlyRootFilesystem)" >&2
    echo "[disk_hog] Only /dev/shm is writable, which is tmpfs — no block I/O possible." >&2
    exit 1
fi

cleanup() { rm -f "$TMPFILE"; }
trap cleanup EXIT INT TERM

echo "[disk_hog] writing to ${TMPFILE} for ${DURATION}s"
end_time=$(( $(date +%s) + DURATION ))
while [ "$(date +%s)" -lt "$end_time" ]; do
    # 8M per iteration: enough I/O to see in fs counters without huge page cache use
    dd if=/dev/urandom of="$TMPFILE" bs=1M count=8 conv=fdatasync 2>/dev/null \
        || dd if=/dev/urandom of="$TMPFILE" bs=1M count=8 2>/dev/null
    rm -f "$TMPFILE"
done

echo "[disk_hog] done"
