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

The LAN8720's RMII **REF_CLK output occasionally skips one 50 MHz
period under PC→FPGA (RX) traffic**. The FPGA's `rmii` domain is clocked
by that REF_CLK, so the miss stalls the RMII TX serializer for one
clock. The PHY keeps clocking and samples the held TXD/TX_EN twice, so
**2 bits are inserted into the frame at a random offset** and the rest of
the frame is shifted by 2 bits. The PC decodes the shifted ADAT slots as
garbage — the audible distortion.

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

5. **REF_CLK skips edges.** REF_CLK was oversampled with a 150 MHz clock
   (mainpll `clk[4]`, VCO/2). Normal REF_CLK period = 3 monitor cycles
   (2–4 with sampling jitter); under RX traffic the observed maximum was
   **6 cycles** (a 20 ns period stretched to ~40 ns = one missing edge).
   Rate: **0 events/s idle**, **≈0.4–2/s while the PC sends audio**,
   matching the corrupted-frame rate in the same windows (e.g. 5 REF_CLK
   gaps vs 6 corrupted frames; a looser threshold counted 57 candidate
   intervals vs 26 corrupted frames, so the tight `>= 6` count is the
   real signal). The FPGA's own full-rate TX does **not** trigger it —
   only PC→FPGA RX activity does.

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

### 6.5 Reproduce the measurement

```
# the payload shift (needs an optical OUT2->IN1 cable)
sudo venv/bin/python etheradat/bringup_tools/loop_exact.py 10 3200

# REF_CLK edge-interval monitor (bring-up build): 150 MHz oversample
# of REF_CLK, counts intervals >= 6 monitor cycles. A diagnostic build
# adds mainpll clk[4] = VCO/2 and a `ref_gap` UART value; see the
# discussion in this file's history. 0/s idle, ~0.4-2/s with RX traffic.
```

