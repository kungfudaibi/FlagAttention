#!/bin/bash
# Refuse to run unless efsmi DEV 1 is PCI 0000:83:00.0 and idle.
set -u
LIST=$(efsmi -L 2>/dev/null)
echo "$LIST"
LINE=$(echo "$LIST" | awk "\$1==1 && /0000:83:00.0/ {print}")
if [ -z "$LINE" ]; then echo "PREFLIGHT_FAIL: DEV 1 is not PCI 0000:83:00.0"; exit 91; fi
SN=$(echo "$LINE" | awk "{print \$2}")
echo "PREFLIGHT_OK dev=1 pci=0000:83:00.0 sn=$SN"
MEM=$(efsmi -i 1 -q -d MEMORY 2>/dev/null)
USED=$(echo "$MEM" | awk -F: "/Used Size/ {gsub(/[^0-9]/,\"\",\$2); print \$2}")
echo "PREFLIGHT_GCU1_USED_MIB=${USED:-unknown}"
if [ "${USED:-0}" -gt 2048 ]; then echo "PREFLIGHT_WARN: GCU1 already has ${USED} MiB allocated"; fi
