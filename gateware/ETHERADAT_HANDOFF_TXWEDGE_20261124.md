# EtherADAT — Handoff: FPGA TX path wedges after a link disturbance (2026-11-24)

**Status: OPEN.** This is the next bug after the ADAT-input fix (commit
`b8e9548`). Both ADAT directions are verified working; the failure is the
Ethernet **device→PC TX direction latching up** until the FPGA is reflashed.

Read this whole file before touching hardware. Context/history:
`ETHERADAT_HANDOFF_INPUT_20261123.md` and `ETHERADAT_STATUS_20261123.md`.

---

## 1. TL;DR

| Item | State |
|------|-------|
| ADAT OUT (PC→Ethernet→FPGA→ADAT OUT1→Audient) | verified (1 kHz) |
| ADAT IN (OUT2→IN1→FPGA→Ethernet→PC) | verified (2 kHz), fixed in `b8e9548` |
| Ethernet TX (device→PC audio frames) | **wedges after a link disturbance** |
| Recovery today | **reflash** `gateware/build/rxfix6.sof` (JTAG, volatile) |
| Goal of this task | automatic recovery (no reflash) |

The wedge is in the **RMII-domain TX sender**, not the ADAT path: the ADAT
receiver keeps decoding normally while the TX side stops draining the TX
block RAM.

---

## 2. Exact signature (how to recognise it)

UART (`etheradat/bringup_tools/uart_rates.py`) in the wedged state (real
capture 2026-11-24):

```
tx_frames    2319 -> 2319     0/s      # frozen: no audio frames reach the MAC
build        2135 -> 2135     0/s      # frozen: builder produces no new block
tx_drop      5288 -> 14097   3204/s    # builder drops EVERY block
c_rx_fdone   46252 -> 47309  48048/s   # ADAT receiver still decoding 48 kHz frames
c_rx_wrap    7453 -> 16262    3204/s   # collector still wraps at the block rate
c_rx_finc    0                         # no incomplete frames
```

Meaning: the builder's `in_flight == 2` (both TX block-RAM buffers still
unconsumed), so it takes the drop path every ring wrap. The RMII sender is
not incrementing its consume counter → it is stuck waiting for the MAC.

PC side: the `adatnet` driver's `eth_rx_frames` freezes and `frames_copied`
stops growing (it only counts audio frames, so this is expected when no
frames are sent). `link` in `/proc/asound/card1/adatnet` may still read
`up` — do not mistake the frozen counter for a driver bug.

**Do not confuse with the benign look-alike:** a dead RMII REF_CLK gives the
same `build` stuck + `tx_drop ≈ 3200/s` signature (and `tx_frames =
hello_tx = 0`), but that is expected when the PHY is not clocking. The wedge
here happens with a live REF_CLK and a live link.

---

## 3. Reproduction

- **Reliable:** `venv/bin/python etheradat/bringup_tools/loop_full.py`
  (concurrent `arecord -c 24 -D hw:1,0` + `aplay -D hw:1,0`). It wedged on
  2026-11-23/24. It plays `tone16.wav` while capturing AdatNet + Audient.
- Also reported after link drops / NIC re-enumeration / USB changes
  (`ETHERADAT_HANDOFF_INPUT_20261123.md` §10: "FPGA TX wedges on Ethernet
  link disturbance").
- `loop_sniff_all.py` (passive `AF_PACKET` sniff, no `arecord`) does **not**
  wedge it.
- Recovery: reflash the bitstream. Reprocedure:
  `ETHERADAT_HANDOFF_INPUT_20261123.md` §8.

---

## 4. TX architecture / where it can latch

Data path (device→PC):

```
builder (fast) --TX block RAM 2x1440 (ping-pong)--> prod_cnt (gray) -> rmii
bridge TX FSM (rmii) --LiteEth MAC user sink--> liteeth_etheradat_mac.v
   -> tx_padding/CRC/preamble/gap -> RMII serializer -> LAN8720
```

The consume handshake that can hang:

- `etheradat/bridge.py:330`  `tx_consume = tx_fsm_valid & self.mac_tx_ready`
- `bridge.py:331-345`  advances `tx_byte`; on the last byte sets
  `tx_sending=0`, increments `tx_cons_cnt`, and emits gray `tx_cons_gray`.
  **`tx_cons_cnt` only advances when a full frame completes.**
- `bridge.py:346-352`  starts `tx_sending` when
  `have_tx_work & ~hello_active & ~discovery.hello_valid`.
- Builder drop condition: `etheradat/audio_payload.py` — `in_flight =
  prod_cnt - cons_bin`; `in_flight == 2` → `drop_hold` (=
  `tx_drops_out`). So a stalled sender shows up as `tx_drop ≈ 3200/s`.

If `mac_tx_ready` never asserts, `tx_sending` stays 1 forever, `tx_byte`
never reaches the last byte, `tx_cons_cnt` never increments → every builder
block is dropped.

Where `mac_tx_ready` comes from:

- `etheradat/liteeth/liteeth_etheradat_mac.v:520`
  `user_sink_ready = sink_ready`; `:970` `sink_ready = tx_padding_sink_ready`.
- The MAC TX pipeline contains an IFG + clock-crossing reset inserter:
  `:556` `cdc_rst = txdatapath_ext_rst_to`; `:557` `tx_gap_reset = cdc_rst`;
  the `txdatapath_resetinserter_state`/`tx_gap_counter` FSM is at `:938-961`.
- The MAC reset inputs come only from `sys_rst` / `eth_tx_rst`, both wired
  to `ResetSignal("rmii")` in `etheradat_interface.py` (the
  `Instance("liteeth_etheradat_mac", ...)` block). There is **no runtime
  reset** of the TX path.
- The `rmii` domain reset is a power-on counter plus `rst_in`, and
  `rst_in` is tied to **0** in the top (`rmii_domain.rst_in.eq(0)`):
  `bridge.py:59-93` (`RMIIClockDomain`).

Note: in this instantiation `i_sys_clk` and `i_eth_tx_clk` are **the same
net** (both the RMII REF_CLK), so there is no real asynchronous crossing —
which makes a permanently-stuck `sink_ready` more surprising and points at
the reset-inserter/IFG FSM latching mid-frame.

Second latching candidate — the hello arbiter (`bridge.py:306-329`):

```
hello_active keeps a hello owning the MAC until its last byte.
safety:  with m.If(hello_active & ~discovery.hello_valid): hello_active.eq(0)
hello_ready = mac_tx_ready & hello_grant
```

If `discovery.hello_valid` is stuck high (the hello stream stalled waiting
for `mac_tx_ready`), the weak safety clear never fires, `hello_active`
stays 1, and audio never resumes. Same upstream trigger (a stuck MAC), but
worth confirming separately because it also starves TX by itself.

---

## 5. Hypotheses (ranked)

1. **`mac_tx_ready` (MAC user sink) stuck low** after a link/PHY
   disturbance: the LiteEth TX IFG/CDC reset-inserter (or the internal
   async FIFO handshake) latches mid-frame and never re-asserts `ready`.
   Nothing resets it at runtime → `tx_sending` waits forever.
2. **`hello_active` / `discovery.hello_valid` deadlock** (arbiter): a stuck
   hello keeps ownership of the MAC and starves audio; the
   `hello_active & ~hello_valid` safety only helps if `hello_valid` drops.
3. **REF_CLK glitch/stop**: if REF_CLK pauses (PHY power/strap glitch,
   link renegotiation), the rmii/eth_tx logic stops mid-frame; on return the
   MAC TX FSM is left mid-frame with no reset.
4. Not the ADAT receiver, not the builder trigger, not the driver: those are
   healthy in the wedge (`c_rx_*` normal; the drop is a *symptom*).

---

## 6. Instrumentation plan (do this first)

Add rmii→sys UART taps exactly like the existing RX taps in
`etheradat_interface.py` (`level`, `level_sig`, `_pulse_to`, `_count`):

- `mac_tx_ready` level, plus a **wedge watchdog**: sticky flag when
  `tx_sending` has been high without `mac_tx_ready` for more than ~2 frame
  periods (~625 µs / ~31k rmii cycles). Latch `tx_byte` at the trip.
- `tx_sending` level; `hello_active` level; `discovery.hello_valid` level.
- `tx_cons_cnt` and `tx_prod_val` (or `in_flight`) levels.

Reproduce with `loop_full.py`, then read the UART. The decisive question:
**is `mac_tx_ready` low while `tx_sending` is stuck, and is `hello_active`
high?** That separates hypothesis 1 from 2. (The existing TX taps —
`sink_*`, `mac_*`, `pin_*` lengths/counters — are already in the dump and
confirm whether any bytes reach the MAC/serializer after the wedge.)

Cheap first check without new logic: after a wedge, reflash and see if it
recovers (it does) — consistent with a latch that only a reset clears.

---

## 7. Fix options (goal: automatic recovery)

**A. MAC TX watchdog + reset (preferred).** If `tx_sending` is stuck without
`mac_tx_ready` for > a few frame periods, abort the frame
(`tx_sending=0`, clear `hello_active`) **and pulse a MAC TX reset** for a
few cycles (add a dedicated `mac_tx_rst` input to the blackbox, or drive a
small reset controller onto `sys_rst`/`eth_tx_rst`). The PC driver already
counts gaps/errors (`eth_rx_gaps`, `eth_rx_errors`) and resyncs. Verify the
generated MAC honours `sys_rst`/`eth_tx_rst` mid-stream (it resets the
rmiitxshim and the tx datapath reset-inserter). This addresses the actual
latch.

**B. REF_CLK activity monitor → macro reset.** Count REF_CLK edges in a
free-running (reset-less) counter; if they stop for T, assert
`reset_out`; hold reset briefly after they return. Covers a PHY/link clock
glitch. `RMIIClockDomain` currently only has the POR counter — add this
there.

**C. Arbiter hardening (defense in depth).** Bound `hello_active`'s
lifetime: drop it after N cycles without `hello_ready`, so a stuck hello
cannot starve audio forever.

**D. Sender-level skip (fallback).** On a frame timeout, drop `tx_sending`
and increment `tx_cons_cnt` so the builder keeps producing; the PC sees a
gap. Simpler but leaves the MAC possibly mid-frame — combine with A.

Preferred: **A + B**, with **C** as a cheap extra. Consider whether the
generated MAC could avoid the CDC entirely (`gen_etheradat_mac.py`; needs
the scratch venv — see the input handoff §2) since `sys_clk == eth_tx_clk`
here.

---

## 8. Notes / gotchas

- Recovery today: reflash. The working ADAT-fix bitstream is
  `gateware/build/rxfix6.sof`, checksum `0x01503569` (persistent mount;
  `build/` is gitignored). FPGA config is volatile — a board power cycle
  needs a reflash.
- A wedge freezes the driver's `eth_rx_frames`; that is not a driver bug.
  `eth_rx_errors`/`eth_rx_gaps` also jump around a link flap.
- The wedge leaves the ADAT receiver healthy — use `c_rx_fdone ≈ 48k/s`
  (UART) to confirm only the Ethernet TX side is stuck.
- `loop_full.py` can wedge the link; prefer `loop_sniff_all.py` for
  RX-direction checks.
- Build/flash procedure: `ETHERADAT_HANDOFF_INPUT_20261123.md` §8.
- Keep wire shapes in mind (`PROTOCOL.md`): aborting a frame mid-stream is
  a malformed frame; prefer resetting the MAC TX path and letting the
  driver drop/resync.

---

## 9. Files

- `etheradat/bridge.py` — TX FSM + hello arbiter (`~240-357`),
  `RMIIClockDomain` (`~59-93`).
- `etheradat/liteeth/liteeth_etheradat_mac.v` — `user_sink_ready`
  (`:520`), `sink_ready` (`:970`), `cdc_rst`/`tx_gap_reset` (`:556-557`),
  reset-inserter FSM (`:938-961`), `eth_tx_rst` reset (`:1688`).
- `etheradat/liteeth/gen_etheradat_mac.py` — MAC generator (scratch venv).
- `etheradat/etheradat_interface.py` — MAC `Instance` reset wiring, UART
  taps.
- `etheradat/audio_payload.py` — builder drop condition / `tx_drops_out`.
- `etheradat/bringup_tools/` — `loop_full.py`, `loop_sniff_all.py`,
  `parse_eduart.py`, `uart_rates.py`.
