# EtherADAT — sample-exactness finding (2026-11-24)

Follow-up to `ETHERADAT_HANDOFF_INPUT_20261123.md` (ADAT IN fixed). Both
directions carry audio, but the user reports the sound is **distorted**.
This note records why, and adds a hardware test that actually checks
sample fidelity instead of only "a tone is present".

## 1. Do the existing tests check exactness?

- **Simulation unit tests do** check bit-exactness:
  - `test_adat_frame_codec.py`: ADATTransmitter -> wire -> ADATReceiver ->
    InputFrameCollector must reproduce the reference 32-byte frames
    byte-for-byte.
  - `test_bridge.py`: the TX payload slots, the RX->ADAT output samples and
    the full ADAT IN -> Ethernet -> ADAT OUT loop are compared to the
    reference encoder/decoder exactly.
- **But** the TX byte-exact tests were **failing in simulation** (3 of 7
  tests) because `bridge.py` compensated for a *fit-specific* block-RAM
  read latency: the address was tuned so the Cyclone IV fit is byte-exact,
  which makes the Amaranth model one byte off. Hardware was authoritative
  and correct; the exactness guard was not green, so a regression there
  could go unnoticed. `test_free_run_silent_blocks` failed with
  `b'\x00\x00!\x84...'` (one byte inserted).
- **The hardware bring-up tools only check presence**: `loop_sniff_all.py`
  and `loop_full.py` measure RMS and the FFT peak of a tone. They cannot
  see a corrupted sample or a shifted frame.

## 2. New tool: `etheradat/bringup_tools/loop_exact.py`

Bypasses ALSA and the kernel driver: it writes AdatNet RX frames straight
to the NIC (AF_PACKET) with a self-describing 24-bit codeword per channel
(0xA5 marker, 4-bit frame index, 4-bit sample index, 3-bit channel, fixed
tail), loops OUT2 -> IN1, sniffs the FPGA TX frames, and requires every
returned 32-byte slot to be byte-identical to a sent one (and every
decoded sample to match its codeword). Also checks the unsynced inputs,
which must be the exact `EMPTY_USER_FRAME`.

    sudo venv/bin/python etheradat/bringup_tools/loop_exact.py 5 3200

## 3. Result (bitstream `0x01503569`, original known-good build)

| send rate | input0 slots | not byte-exact | unsynced EMPTY corrupted |
|-----------|--------------|----------------|--------------------------|
| 50 /s     | 44430        | 0              | 0                        |
| 400 /s    | 82965        | 0              | 0                        |
| 800 /s    | 77910        | 0              | 0                        |
| 1600 /s   | 109875       | 2              | 0                        |
| 3200 /s   | 188985       | 40             | 81 / 377889              |

At the **nominal 3200 blocks/s** about **0.02 %** of TX payload slots are
corrupted (~40-70 per 5 s run). That is the audible distortion. It is not
constant: some runs are clean, and it scales with Ethernet RX traffic
(flooding the link with 1400-byte junk frames at 4000/s also reproduced
it while the audio was sent at only 50/s).

Corrupt slots are **non-canonical** (their ADAT marker bits are wrong),
i.e. the corruption is in the **device->PC TX payload byte stream**
(`builder -> TX block RAM -> bridge -> MAC`), not in the ADAT receiver:
`c_rx_fdone = 48048/s`, `c_rx_finc = 0`, `sync0 = 1` throughout. It hits
the synced input *and* the constant unsynced `EMPTY` slots, so it is the
shared TX path, not the ADAT input.

## 4. What was tried and reverted

- **Latency-agnostic 3-cycle block-RAM fetch (no FIFO):** made the
  simulation byte-exact, but the per-byte `valid` gaps made the LiteEth
  MAC's TX buffer underrun mid-frame; unsynced slots became corrupted on
  hardware too. Rejected.
- **Hand-rolled 4-entry prefetch FIFO:** made all three sim exactness
  tests pass, and one 3200/s hardware run was bit-exact, but at 50/s the
  MAC still saw gaps (the 1-byte-per-3-cycles fetch is slower than the
  sink wants) and 1/3 of slots corrupted. Rejected.
- **Reverted to the committed `bridge.py`** and reflashed the original
  bitstream (`0x01503569`). The board is in the state described above.

## 5. Likely root cause / next step

> **SUPERSEDED — see §6: the cause was found and measured. The corruption
> is a hardware REF_CLK edge skip under RX traffic, not a payload-logic
> or driver bug. Nothing below needs to be built.**

The LiteEth MAC blackbox hosts both the user TX sink and the RX source on
the same clock; the TX IFG/CDC reset inserter is the suspect for the
known TX wedge (`ETHERADAT_HANDOFF_TXWEDGE_20261124.md`). Earlier ideas
(regenerate the MAC, add an external FIFO) are not needed: both were
tried during this investigation and did not help (§6.4).

## 6. ROOT CAUSE — measured 2026-11-24

### 6.1 Summary

The LAN8720's RMII **REF_CLK output occasionally stretches a 20 ns
period to 32–36 ns under PC→FPGA (RX) traffic** (measured at 250 MHz,
§6.6: ~20/s loaded, 0/s idle). The FPGA's `rmii` domain is clocked by
that REF_CLK, so the late edge delays the RMII TX serializer by
12–16 ns. Depending on where the stretched interval falls relative to
the PHY's own clean internal clock, the PHY can then sample the held
TXD/TX_EN twice, so **2 bits are inserted into the frame at a random
offset** and the rest of the frame is shifted by 2 bits. The PC decodes
the shifted ADAT slots as garbage — the audible distortion.

It is a hardware clock-integrity issue. It is **not** the kernel driver,
not the ADAT receiver, and not the bridge/MAC payload logic.

### 6.2 Evidence chain

1. **The kernel driver is not involved.** `lsmod` shows `adatnet`
   unloaded; `loop_exact.py` uses `AF_PACKET` directly. The wedge states
   seen during this session were in experimental FPGA FIFOs (since
   reverted), and the documented TX wedge is FPGA-internal
   (`mac_tx_ready`/IFG latch) as the TXWEDGE note says.

2. **The host NIC does not drop bad-FCS frames here.** A temporary
   diagnostic corrupted one byte *after* the MAC's CRC inserter in every
   frame. All frames were still delivered and `rx_errors`/`rx_crc_errors`
   stayed at 0, so a bad FCS cannot be used to rule out post-CRC
   corruption (the earlier "FCS is valid" reasoning was wrong).

3. **The corruption is a 2-bit serializer slip, not a data-path error.**
   From a random byte `p0` to the end of the frame the payload equals the
   true block stream shifted by 2 bits:
   `out[i] = (in[i] << 2) | (in[i-1] >> 6)`.
   On the *unsynchronized* input slots (whose true value is the constant
   `EMPTY_USER_FRAME`) this holds **exactly, including the carry**, i.e.
   the two inserted bits are a repeat of the preceding dibit — the
   signature of the serializer holding for one clock, not of a PHY idle
   insertion or a valid/data gap.

4. **The FPGA byte stream and TX_EN are continuous.** Instrumented over a
   5.75 s full-rate run: the shim's `sink_valid` falling-edge count was
   exactly the frame count (`valid_falls == tx_frames`), and TX_EN
   falling edges matched frames (`txen_falls == tx_frames + 1`). There
   were **no mid-frame valid gaps and no extra TX_EN deassertions**.

5. **REF_CLK intervals stretch under load.** First seen with a 150 MHz
   oversampler (mainpll `clk[4]`, VCO/2): normal period 3 monitor
   cycles, under RX traffic up to 6. Re-measured 2026-11-24 at 250 MHz
   with a validated measurement chain (§6.6): **0 events/s idle, ~20/s
   stretched intervals (32–36 ns) while the PC sends 3200 frames/s; no
   full 40 ns "missing edge" was seen at this resolution**. The FPGA's
   own full-rate TX does **not** trigger it — only PC→FPGA RX activity
   does.

6. **The FPGA cannot clean REF_CLK in gateware.** REF_CLK is on pin
   `J_2:20`, which is **not connected to any PLL `INCLK`** on the
   EP4CGX150: trying an `ALTPLL` fed by REF_CLK fails the fit with
   `Error (176554): ... pin rmii_0__ref_clk__io ... not connected to port
   type INCLK of any PLL`. So a PLL cannot be used to coast through the
   skipped edge.

### 6.3 The fix (hardware — cannot be done from the VM)

Make the **FPGA the RMII REF_CLK master** so both ends use one clean
clock and an edge cannot be skipped on only one side:

1. Strap the LAN8720 module for **REF_CLK-IN** (`nINTSEL`, pin 2) and
   confirm the module's REF_CLK pin is wired to its external-clock input
   (datasheet pin 14 is `nINT/REFCLKO` or `XTAL1` depending on variant —
   see `etheradat/README.md` §"REF_CLK-Richtung"). This is a board-level
   strap/jumper change.
2. Build with `--ref-clk-mode output`
   (`generate_quartus_files_etheradat.py ... --ref-clk-mode output`).
   The FPGA then drives REF_CLK from the board's clean 50 MHz oscillator
   and clocks the whole `rmii` domain from that same net.
3. Verify: `sudo venv/bin/python etheradat/bringup_tools/loop_exact.py 10 3200`
   must print `RESULT: BIT-EXACT` on several consecutive runs.
   `uart_rates.py` should show `build`/`tx_frames` ≈ 3200/s, `tx_drop` 0.

Alternative fixes if the module cannot be re-strapped:
- Repair the module's REF_CLK/power/ground integrity (the same module
  also fails MDIO link detection: `status0 link=0 phy_found=0`, which
  points at module power/connector trouble — see
  `ETHERADAT_STATUS_20261123.md`), or
- Rework REF_CLK onto a PLL-capable FPGA input pin and add a cleanup
  PLL clocking the `rmii` domain.

No kernel-driver change is required or useful: a clock-input glitch is
invisible to the driver except as bad samples, and no driver action can
prevent it.

### 6.4 What was tried during this investigation (all reverted)

- **Shim prefetch FIFO** (fill the serializer ahead from a depth-8 FIFO):
  the MAC's byte stream is already continuous (§6.2.4), so there was no
  bubble to absorb; changing the shim's `sink_ready` pacing instead
  exposed the bridge block-RAM read latency and starved frames.
- **Bridge payload read FIFO** (latency-agnostic block-RAM read): its
  bridge-level unit tests pass in simulation, but on hardware the extra
  handshake/FIFO state wedged or under-ran under load (further
  instrumentation needed). Not the cause; reverted.

The board is back on the committed baseline bitstream (`0x01503569`
logic, rebuilt from `HEAD`); it still shows the documented ~0.02 %
corruption until the REF_CLK fix above is applied.

### 6.5 Schmitt-trigger input attempt (negative — hardware option)

Goal: reject the crosstalk-induced excursions on the REF_CLK line with
input hysteresis, without changing the board.

On Cyclone IV the Schmitt trigger is **not** a separate assignment
(`set_instance_assignment -name SCHMITT_TRIGGER ...` is rejected as an
illegal assignment name). It is only exposed as an **I/O standard**, and
the only names Quartus recognizes are:

- `3.3V Schmitt Trigger Input`
- `3.3 V Schmitt Trigger` (standalone-input variant)

Both were applied to `rmii_0__ref_clk__io` (PIN_B18, input mode) in a
build. The fitter accepts the names (the error changes from 169216
"not a valid I/O standard name" to 169059) but then refuses them:

```
Error (169059): I/O standard assignment 3.3 V Schmitt Trigger to pin
  rmii_0__ref_clk__io is not supported by device
Error (171000): Can't fit design in device
```

So the EP4CGX150 DF27 (Cyclone IV GX) has no user-selectable Schmitt
trigger I/O standard. The generator patch was reverted; the clean
input-mode baseline was rebuilt (fitter successful, B18 = input,
`3.3-V LVCMOS`, used as clock). This closes the cheap in-FPGA option.

Remaining options (all board-level), in increasing effort:

1. **Signal integrity on the REF_CLK wire**: short wire with a ground
   return / twisted pair, routed away from RXD0/RXD1/CRS_DV/TXD, and a
   ~33–68 Ω series resistor at the source. The module's REF_CLK copy is
   the only signal that glitches, under RX traffic only, which is
   consistent with crosstalk/ringing on that one line.
2. **External Schmitt buffer** (e.g. 74AHC1G14 / 74LVC1G17) directly at
   the FPGA REF_CLK pin. Adds ~2–5 ns of clock delay, which shifts all
   RMII TX transitions later within the 20 ns period; that direction is
   the safe one (more setup margin, still inside the PHY hold window),
   but keep the part fast and the wire short.
3. **Make the FPGA the clock master** (fixes it completely): the module
   oscillator drives both the PHY clock input and the header REF_CLK
   pin. Remove/disable that oscillator and drive the same net from the
   FPGA REF_CLK output (`--ref-clk-mode output`); the FPGA then clocks
   the `rmii` domain from the same net it drives, so there is no
   second copy left to glitch. Requires desoldering the oscillator can
   or cutting its output trace.
4. **Rework REF_CLK onto a PLL-capable input pin** and clean it with a
   PLL. No J_2 pin is INCLK-capable (all 54 fail with error 176554);
   this needs a pin on another connector and a new pin assignment.

### 6.6 250 MHz REF_CLK monitor (validated, 2026-11-24)

The 150 MHz numbers were threshold-sensitive, so REF_CLK is now
oversampled by a **dedicated ALTPLL at 250 MHz** (VCO 750 MHz / 3, one
free GPLL; 4 ns per tick) and the monitor is part of the interface
(`EtherADATInterface`, input mode only; UART values 66..73, build marker
`0xCA35`, parser `NV = 74`).

Measurement chain and pitfalls (both were hit and fixed):

- The edge must be taken from the **`FFSynchronizer` output plus one
  history FF**. Deriving it from the first sampling FF (hand-rolled
  2-bit shift register) produced a bogus 55k/s "short gap" storm at idle
  and spurious "long" gaps under load — a metastable first stage loses
  and fabricates edges.
- A 2-FF synchronizer adds **0..2 ticks of edge jitter**: at 250 MHz the
  normal interval is 5 ticks and 3..7 is still normal. The bins are
  chosen so jitter cannot masquerade as a defect: `short <=2` (jitter),
  `stretch 8..9` (32–36 ns), `skip 10..12` (40–48 ns = one missing
  edge), `vlong >=13`. Only stretch/skip/vlong exceed the jitter bound.

Control experiment: `--mon-source clk50` monitors the board's own 50 MHz
oscillator, which also feeds the monitor PLL (source and sampler
synchronous). Under the same 3200 frames/s load this control build
reports **zero** stretch/skip/vlong events, so the monitor itself cannot
fabricate them.

REF_CLK results (marker `0xCA35`, `loop_exact.py 12 3200`):

| state | c_ref_short (≤8 ns) | c_ref_stretch (32–36 ns) | c_ref_skip (40–48 ns) | vlong |
|-------|--------------------|--------------------------|-----------------------|-------|
| idle  | 0/s                | **0/s**                  | 0/s                   | 0/s   |
| load  | ~4/s               | **~20/s**                | 0/s                   | 0/s   |

The last events were gap = 9 ticks (36 ns) with the previous interval
4 ticks, monitor PLL locked. So the clock does not lose a whole cycle;
its **edge arrives 12–16 ns late ~20 times/s under RX traffic**. That is
enough for the phase between the FPGA serializer and the PHY's internal
clean clock to cross a sampling boundary, producing the 2-bit slips.
The effect is RX-activity-correlated (crosstalk on the REF_CLK line),
which is why hardware mitigation (signal integrity, external buffer,
FPGA-as-master) is the fix — see §6.3.

### 6.7 Reproduce the measurement

```
# the payload shift (needs an optical OUT2->IN1 cable)
sudo venv/bin/python etheradat/bringup_tools/loop_exact.py 10 3200

# 250 MHz REF_CLK monitor: build (input mode is the default and
# includes the monitor), flash, then read the UART as usual:
LUNA_PLATFORM=platforms:ADATFaceCycloneIV_EP4CGX150 \
    venv/bin/python etheradat/scripts/generate_quartus_files_etheradat.py \
    /tmp/rebuild_mon
sudo stty -F /dev/ttyUSB0 115200 cs8 -cstopb -parenb raw -echo
timeout 8 sudo cat /dev/ttyUSB0 >| /tmp/uart_load.bin   # while loop_exact runs
venv/bin/python etheradat/bringup_tools/uart_rates.py < /tmp/uart_load.bin

# control run on the same PLL's own source (must show zero events):
#   ... generate_quartus_files_etheradat.py /tmp/rebuild_ctrl --mon-source clk50

# Values: c_ref_alive heartbeat; c_ref_short/stretch/skip/vlong bins;
# c_ref_prev (gap before the last event); c_ref_last (gap | kind |
# PLL lock); marker 0xCA35.
```


## 7. Cyclone V (5CEFA2F23C8) attempt — 2026-11-24

Goal: check whether the Cyclone V core board handles the REF_CLK edge
problem better, and whether it offers the Schmitt-trigger or cleanup-PLL
option that the Cyclone IV lacks.

### 7.1 Build

The top does **not** fit the A2 as-is: 1004 LABs needed / 943, 105 % of
the ALMs; with the 250 MHz monitor PLL it also fails to place the PLL
(`Error 14566/175001`, FRACTIONALPLL_X0_Y1_N0 conflict with mainpll).

With `--no-debug-taps --mon-source none` and the Cyclone V platform
now forcing `OPTIMIZATION_MODE "AGGRESSIVE AREA"` +
`PHYSICAL_SYNTHESIS_EFFORT NORMAL` (added to `platforms.py`), it fits:
**8221 / 9430 ALMs (87 %)**, 15887 registers, all setup slacks positive
(min 1.638 ns). The monitor is not included in this build (PLL budget).

### 7.2 Bring-up result: RX works, TX framing is broken

- UART dump works; FPGA RX side receives the full PC stream
  (`rx_frames` = 3200/s, `rx_err` = 0), so PHY, cable and the FPGA's
  RX/REF_CLK sampling are fine on Cyclone V.
- The MAC TX side counts `tx_frames` ≈ 3256/s, but the PC receives
  almost nothing: `loop_exact` sniffs 0 frames at 50/s and 400/s (3
  frames in 10 s at 3200/s). `tcpdump` *does* see the frames, but with
  **variable lengths 1468–1472 bytes instead of 1462** (broadcast,
  since the PC MAC is not linked). `rx_errors` stays 0 — the AX88179
  drops the malformed frames internally.
- `--tx-negedge` does not change it. This is a Cyclone V **porting
  issue in the TX path**, most likely the documented fit-specific
  block-RAM read behaviour that `bridge.py` compensates for
  (§6.4/§5): on the M10K the frame on the wire gets 6–10 extra bytes.
  It blocks any corruption comparison on this board and needs its own
  fix (not attempted here).

### 7.3 Schmitt trigger: not available either

Both `3.3V Schmitt Trigger Input` and `3.3 V Schmitt Trigger` on
`rmii_0__ref_clk__io` (PIN_AA19) are rejected by the fitter:

```
Error (169059): I/O standard assignment ... is not supported by device
```

Same result as the Cyclone IV (§6.5) — the Cyclone V A2 has no usable
Schmitt-trigger I/O standard for this pin.

### 7.4 Cleanup PLL: REF_CLK pin is not PLL-capable

A minimal probe (an `ALTPLL` fed directly by the REF_CLK pin, plus a
counter on a debug pin) fails with:

```
Error (177020): The PLL reference clock was not placed in a dedicated
  input pin that can reach the fractional PLL
```

So AA19 cannot drive a PLL on the 5CEFA2 either. The "rework REF_CLK
onto a PLL-capable pin" option still requires a *different* pin (a
dedicated CLK input), not just a different core board.

### 7.5 Conclusion

Cyclone V offers neither the Schmitt trigger nor a cleanup PLL for this
pin, and the EtherADAT top needs a TX-path port before it can even be
compared on the same loopback test. A family change alone does not
address the module's REF_CLK line, which is the actual problem (§6).
The practical fixes remain the hardware measures in §6.3/§6.5.

Reproduce the probe:

```bash
# the probe script is kept in the conversation; the essential check is
# an ALTPLL with i_inclk = the rmii ref_clk pin.  Build with:
LUNA_PLATFORM=platforms:ADATFaceCycloneV \
    venv/bin/python etheradat/scripts/generate_quartus_files_etheradat.py \
    /tmp/rebuild_cv --mon-source none --no-debug-taps
```
