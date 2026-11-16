#!/usr/bin/env bash
# Testet EINE .sof-Datei: volatil per JTAG laden, frische USB-Enumeration erzwingen,
# ALSA-Status + Stabilität prüfen. Überschreibt NIEMALS den Flash.
# Usage: test_one.sh <label> <pfad-zur-sof>
set -u
LABEL="$1"
SOF="$2"
QBIN=/opt/altera_lite/25.1std/quartus/bin
export PATH="$QBIN:$PATH"
LOGDIR=/mnt/adat-usb2-audio-interface/gateware/firmware_test_logs_cycloneV
LOG="$LOGDIR/$LABEL.log"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

log "===== TEST $LABEL ====="
log "sof=$SOF"
DEVNUM_BEFORE=$(grep -h devnum /sys/bus/usb/devices/*/devnum 2>/dev/null | grep -c . || true)

log "--- JTAG-Load (volatil, 'p') ---"
quartus_pgm -m jtag -o "p;$SOF" >> "$LOG" 2>&1
PGMRC=$?
log "quartus_pgm rc=$PGMRC"
if [ $PGMRC -ne 0 ]; then
    log "RESULT: LOAD_FAILED"
    exit 1
fi

sleep 5
log "--- USB-Reset erzwingen ---"
log "usb_reset: $(sudo python3 /mnt/adat-usb2-audio-interface/gateware/tools/firmware_test/usb_reset.py 2>&1 || true)"
sleep 20

log "--- Enumeration-State ---"
log "lsusb_ada1: $(lsusb | grep -c ada1) Treffer"
cat /proc/asound/cards >> "$LOG" 2>&1
log "--- aplay -l ---"
aplay -l >> "$LOG" 2>&1
log "--- arecord -l ---"
arecord -l >> "$LOG" 2>&1
log "--- dmesg (letzte Zeilen zu usb 1-2) ---"
sudo dmesg -T | grep -E 'usb 1-2' | tail -15 >> "$LOG" 2>&1

log "--- Stabilitätstest (capture 2s) ---"
STAB=FAIL
for c in 2 8 36 16 4; do
    if arecord -D hw:0,0 -c "$c" -r 48000 -f S32_LE -d 2 -t raw /dev/null >> "$LOG" 2>&1; then
        STAB="PASS(ch=$c)"
        break
    else
        log "  capture c=$c fehlgeschlagen (rc=$?)"
    fi
done
sleep 5
if ! lsusb | grep -q ada1; then
    log "Gerät nach Stabilitätstest VERSCHWUNDEN"
    STAB="$STAB +DISAPPEARED"
fi
log "RESULT: $STAB"
echo "$LABEL => $STAB"
