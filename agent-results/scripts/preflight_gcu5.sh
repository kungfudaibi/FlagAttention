#!/bin/bash
# Refuse to run unless efsmi DEV 5 is PCI 0000:83:00.0, and report usage.
#
# SAGE-003 is pinned to container GCU 5. The device ordinal alone is not proof:
# efsmi is the only authority on which physical card an ordinal maps to, and a
# different container numbers the same PCI address differently (SAGE-002 ran on
# this card as "DEV 1"). Checking the PCI address keeps the guarantee portable.
set -u
EXPECTED_PCI="0000:83:00.0"
EXPECTED_SN="A0D5460520864"
LIST=$(efsmi -L 2>/dev/null)
LINE=$(echo "$LIST" | awk -v want="$EXPECTED_PCI" '$0 ~ want {print}')
if [ -z "$LINE" ]; then
    echo "PREFLIGHT_FAIL: no efsmi device at PCI $EXPECTED_PCI"
    exit 91
fi
DEV=$(echo "$LINE" | awk '{print $1}')
SN=$(echo "$LINE" | awk '{print $2}')
if [ "$DEV" != "5" ]; then
    echo "PREFLIGHT_FAIL: PCI $EXPECTED_PCI is efsmi DEV $DEV, expected DEV 5"
    exit 91
fi
if [ "$SN" != "$EXPECTED_SN" ]; then
    echo "PREFLIGHT_FAIL: PCI $EXPECTED_PCI has serial $SN, expected $EXPECTED_SN"
    exit 91
fi
echo "PREFLIGHT_OK dev=5 pci=$EXPECTED_PCI sn=$SN"
MEM=$(efsmi -i 5 -q -d MEMORY 2>/dev/null)
USED=$(echo "$MEM" | awk -F: '/Used Size/ {gsub(/[^0-9]/,"",$2); print $2}')
echo "PREFLIGHT_GCU5_USED_MIB=${USED:-unknown}"
if [ "${USED:-0}" -gt 1024 ]; then
    echo "PREFLIGHT_WARN: GCU 5 already has ${USED} MiB allocated"
fi
