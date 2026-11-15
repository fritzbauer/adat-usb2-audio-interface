#!/usr/bin/env python3
"""Hardware-in-the-loop audio path test for the ADATface.

Loopback wiring (user-provided):
  - ADATface outputs 1-8 -> Audient iD14 ADAT input, seen there as capture ch 3-10
  - ADATface outputs 9-15 -> ADATface inputs 9-15 (optical loopback)

Test A (output path): play 36ch on ADATface, record 10ch on iD14 (hw:1,0).
  Compare ADATface playback ch 1-8 against iD14 capture ch 3-10.
Test B (input path): play 36ch on ADATface, record 36ch on ADATface (hw:0,0).
  Compare playback ch 9-16 against capture ch 9-16.

Test signal: each channel c (0-based) carries a self-identifying ramp
    v(n, c) = ((n & 0x3FFF) << 9) | (c << 4)
which is 24-bit safe. The device keeps only the upper 24 bits of the
S32_LE word (left-justified), so the channel tag below bit 8 is masked
off; the ramp counter (n << 9) survives and is used to verify sample
continuity. n=0 samples read back as 0 (truncation) -- not a defect.

Baseline of the WORKING flash firmware (adatface_fir6, 2022-06-01),
measured 2026-11-14:
  - data otherwise bit-exact (after 24-bit masking)
  - one deterministic startup hiccup at t=0.042s: +1 duplicated sample
    plus one stale sample (TX path, visible on both capture devices)
  - rare single-sample dup/drop pairs (net drift 0) at ~1-2 s intervals
    on the capture side (iD14 clock resync / ADAT RX path)

PASS criteria: sample continuity holds (allowing dup/drop corrections)
with <= MAX_EVENTS extra/missing samples per channel over the capture,
and no structural corruption (wrong values in sequence).

Run with: gateware/venv/bin/python tools/audio_test/audio_loopback_test.py [A|B|all]
"""
import subprocess
import sys
import time
import os

import numpy as np

SR           = 48000
PLAY_SECONDS = 7.0
REC_SECONDS  = 6.0
PLAY_CH      = 36
TMPDIR       = "/tmp/adat_audio_test"

ADATFACE = "hw:CARD=ADATfaceUSB1,DEV=0"
ID14     = "hw:CARD=iD14,DEV=0"

MAX_EVENTS_PER_S = 3.0   # dup/drop corrections tolerated per channel-second


def gen_playback(nframes: int, nch: int) -> np.ndarray:
    n = np.arange(nframes, dtype=np.int64)
    out = np.zeros((nframes, nch), dtype=np.int32)
    for c in range(nch):
        out[:, c] = (((n & 0x3FFF) << 9) | (c << 4)).astype(np.int32)
    return out


def load_raw(path: str, nch: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    usable = (len(data) // nch) * nch
    return data[:usable].reshape(-1, nch)


def continuity_census(q: np.ndarray, verbose=False, max_print=12):
    """q: one channel, already 24-bit masked. Decode the ramp counter
    (q >> 9) & 0x3FFF and track sample continuity, allowing dup/drop
    corrections. Returns (n_events, n_bad_samples, ok, details).
    ok=False on structural corruption (non-consecutive wrong values)."""
    nz = np.nonzero(q != 0)[0]
    if len(nz) == 0:
        return 0, len(q), False, ["all zero"]
    qs = int(nz[0])
    n = ((q[qs:] >> 9) & 0x3FFF).astype(np.int64)
    # ramp value 0 is truncated to 0 -> invisible; start at n==1
    cstart = np.nonzero(n == 1)[0]
    if len(cstart) == 0:
        return 0, len(q), False, ["ramp counter not found"]
    cstart = int(cstart[0])
    n = n[cstart:]
    L = len(n)
    events, details = 0, []
    bad = 0
    i = 0
    struct_bad = 0
    while i < L - 6:
        if n[i + 1] == (n[i] + 1) % 16384:
            i += 1
            continue
        found = None
        for k in range(2, 6):
            if n[i + k] == (n[i] + 1) % 16384:
                found = k
                break
        if found is not None:
            extra = found - 1
            desc = f"+{extra} extra"
            if found == 3:
                desc += f" stale n={int(n[i + 2])}"
            events += 1
            bad += extra
            if len(details) < max_print:
                details.append((i, desc))
            i += found
        else:
            delta = int((n[i + 1] - n[i]) % 16384)
            if delta <= 64:
                # simple drop of (delta-1) samples (clock slip / FIFO glitch)
                events += 1
                bad += delta - 1
                if len(details) < max_print:
                    details.append((i, f"-{delta-1} dropped"))
                i += 1
            else:
                # big discontinuity: try resync to n[i]+1 within a window
                # (capture-side pauses of a few ms, e.g. USB device resync,
                #  count as drops, not corruption)
                target = (n[i] + 1) % 16384
                hit = None
                for k in range(1, 8192):
                    if i + k < L and n[i + k] == target:
                        hit = k
                        break
                if hit is not None:
                    events += 1
                    bad += hit - 1
                    if len(details) < max_print:
                        details.append((i, f"-{hit-1} dropped"))
                    i += hit
                else:
                    struct_bad += 1
                    if len(details) < max_print:
                        details.append((i, f"CORRUPT delta={delta}"))
                    i += 1
    return events, bad, (struct_bad == 0), details


def report_channel(q_raw: np.ndarray, label: str, seconds: float):
    q = q_raw.astype(np.int64) & ~0xFF
    events, bad, ok, details = continuity_census(q)
    L = max(len(q), 1)
    limit = max(4, int(MAX_EVENTS_PER_S * seconds))
    status = "PASS" if (ok and events <= limit) else ("CORRUPT" if not ok else "EVENTS")
    if status != "PASS":
        pass  # caller aggregates
    rate = events / seconds
    print(f"  {label}: events={events:3d} ({rate:.2f}/s, limit {limit}) bad-samples={bad} "
          f"[{status}]")
    for pos, desc in details[:8]:
        print(f"        t~{pos/48000:8.4f}s: {desc}")
    return status == "PASS"


def report_A(ch_pairs, play, cap, seconds, label="testA"):
    print(f"\n=== {label}: ADATface out 1-8 -> iD14 ch 3-10 (independent clocks) ===")
    ok_all = True
    for pc, cc in ch_pairs:
        ok_all &= report_channel(cap[:, cc], f"play ch {pc+1:2d} -> iD14 ch {cc+1:2d}", seconds)
    return ok_all


def report_B(channels, play, cap, seconds, label="testB"):
    print(f"\n=== {label}: ADATface out -> ADATface in (same clock domain) ===")
    ok_all = True
    for c in channels:
        ok_all &= report_channel(cap[:, c], f"ch {c+1:2d}", seconds)
    return ok_all


def run_arecord(dev, nch, seconds, out_path):
    return subprocess.Popen(
        ["arecord", "-D", dev, "-c", str(nch), "-r", str(SR),
         "-f", "S32_LE", "-d", str(int(seconds)), "-t", "raw", out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def run_aplay(dev, nch, in_path):
    # large buffer: absorbs host (VM) jitter; a small buffer makes the
    # USB OUT packets arrive late and the gateware replays/skips frames
    return subprocess.Popen(
        ["aplay", "--buffer-size=48000", "-D", dev, "-c", str(nch), "-r", str(SR),
         "-f", "S32_LE", "-t", "raw", in_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(TMPDIR, exist_ok=True)
    nframes = int(SR * PLAY_SECONDS)
    play_path = os.path.join(TMPDIR, "play36.raw")
    if not os.path.exists(play_path) or os.path.getsize(play_path) != nframes * PLAY_CH * 4:
        print(f"generating playback signal ({nframes} frames x {PLAY_CH} ch)...")
        gen_playback(nframes, PLAY_CH).tofile(play_path)
    results = {}

    if which in ("A", "all"):
        cap_path = os.path.join(TMPDIR, "cap_iD14.raw")
        if os.path.exists(cap_path):
            os.remove(cap_path)
        print("=== test A: ADATface playback -> iD14 capture ===")
        rec = run_arecord(ID14, 10, REC_SECONDS, cap_path)
        time.sleep(0.4)
        play = run_aplay(ADATFACE, PLAY_CH, play_path)
        rc_r = rec.wait()
        rc_p = play.wait()
        if rc_r != 0 or rc_p != 0:
            print(f"  subprocess failed: arecord rc={rc_r} {rec.stderr.read()!r} "
                  f"aplay rc={rc_p} {play.stderr.read()!r}")
        cap_data = load_raw(cap_path, 10)
        print(f"  captured {len(cap_data)} frames x 10 ch")
        results["A"] = report_A([(i, 2 + i) for i in range(8)], None, cap_data, REC_SECONDS)

    if which in ("B", "all"):
        cap_path = os.path.join(TMPDIR, "cap_adatface.raw")
        if os.path.exists(cap_path):
            os.remove(cap_path)
        print("=== test B: ADATface playback -> ADATface capture (ADAT loopback 9-16) ===")
        rec = run_arecord(ADATFACE, PLAY_CH, REC_SECONDS, cap_path)
        time.sleep(0.4)
        play = run_aplay(ADATFACE, PLAY_CH, play_path)
        rc_r = rec.wait()
        rc_p = play.wait()
        if rc_r != 0 or rc_p != 0:
            print(f"  subprocess failed: arecord rc={rc_r} {rec.stderr.read()!r} "
                  f"aplay rc={rc_p} {play.stderr.read()!r}")
        cap_data = load_raw(cap_path, PLAY_CH)
        print(f"  captured {len(cap_data)} frames x 36 ch")
        results["B"] = report_B(range(8, 16), None, cap_data, REC_SECONDS)

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  test {k}: {'PASS' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
