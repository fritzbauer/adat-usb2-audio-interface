# EtherADAT — Handoff: ADAT-INPUT (RX) fix, context + next steps (2026-11-23)

This is a handoff for a fresh agent picking up **one remaining bug: the ADAT
input direction (ADAT IN -> Ethernet -> PC) delivers silence/garbage**, even
though the ADAT output direction is fully verified. Everything else (build,
flash, driver, full-duplex link, TX frames) is working. Read this whole file
before touching hardware.

See also `gateware/ETHERADAT_STATUS_20261123.md` (full history) and
`gateware/ETHERADAT_PROMPT.md` (task). The committed fix that got us here is
commit `038afcf` ("fix corrupt audio frames on the wire").

---

## 0. RESOLVED (same day, evening) — ADAT IN now works end-to-end

Two independent bugs, both required:

1. **Fast clock was 9x the ADAT bit rate, not 8x.** The shared
   `adat`/`ADATReceiver`/`NRZIDecoder` hardcodes `bit_time = 8` fast cycles
   per ADAT bit. The Cyclone V platform declares `fast_multiplier = 8`
   (fast PLL actual = 98.304 MHz), but the Cyclone IV platforms
   (`ADATFaceCycloneIV`, `..._EP4CGX150`, `ADATFaceCyclone10`) had
   `fast_multiplier = 9`; `car.py` feeds that straight into the fast PLL
   `CLK0_MULTIPLY_BY`, so the fit produced **110.61 MHz** (confirmed in
   `top.fit.rpt`). The receiver "synced" (the lock window is wide) but its
   free-running bit counter ran 12.5 % too fast and lost/duplicated bits,
   decoding EMPTY/garbage. Fixed: `fast_multiplier = 8` on the Cyclone IV
   platforms. Reproduced in simulation: the wire->receiver->collector chain
   decodes exactly at 8x and returns the seeded EMPTY frame at 9x.
   Regression guard: `etheradat/tests/test_platform_clocks.py`.

2. **Block builder never triggered on a synced input.** Once `sync0` became
   solid, `free_run` went to 0 and the builder's wrap trigger
   `Array([wrap0,wrap1,wrap2])[master]` never fired (`build = 0`, while the
   collector ring wrapped at 3200/s and the receiver decoded 48 kHz frames).
   Amaranth lowers `Array-with-Signal-index` to a `casez(master)` with a
   wildcard default; on Cyclone IV / Quartus 25.1 that mux did **not**
   select element 0 even with `master == 0`. Fixed: explicit priority
   AND/OR for the master wrap (and explicit `Mux` for the synced-path input
   byte select, same lowering target).

Verified on hardware (bitstream checksum `0x01503569`) with the UART taps
and the full loop:
- UART: `tx_frames == build ≈ 3200/s`, `sync0 = 1`;
  `c_rx_fdone ≈ 48000/s`, `c_rx_wrap ≈ 3200/s`, `c_rx_finc = 0`;
  `rx_state = 0` (DECODE).
- `loop_sniff_all.py` (OUT2=2 kHz -> IN1): input0 ch0-7 = **2008 Hz**.
- `loop_full.py`: AdatNet capture ch0-7 = **2002 Hz**; Audient ADAT
  ch2-9 = **1002 Hz** (OUT1 path unchanged).

Known, pre-existing test caveat: the `test_bridge.py` *byte-exact payload*
assertions (`test_free_run_silent_blocks`,
`test_tx_frames_match_reference_layout`, full loop) still fail in the
Amaranth sim by one byte — this is the documented sim-vs-fit block-RAM read
offset (§5.2), independent of this fix. `test_discovery`'s
`test_hello_frame_matches_reference` also fails pre-existing/unrelated.


## 1. TL;DR / current state

| Area | Status |
|------|--------|
| Build (VM generate → host synth → JTAG flash) | works (see §8) |
| FPGA running, UART alive, all clock domains | yes |
| TX frames on the wire (1462 B, header + exact `EMPTY_USER_FRAME`) | **verified**, 45/45 slots exact |
| Kernel driver `adatnet` link + free-run decode | **link up**, `hello_rx` rising, `eth_rx_frames` ~3200/s |
| Ethernet link duplex/bandwidth | **100 Mb/s FULL**, NIC on USB2 480 M |
| **ADAT OUT** (PC → Ethernet → FPGA → ADAT OUT1 → Audient iD14) | **VERIFIED**: Audient ADAT ch2-9 shows the 1 kHz tone at full level (rms ≈ 1.5e8) |
| **ADAT IN** (OUT2 → IN1 → FPGA → Ethernet → PC) | **VERIFIED**: input0 ch0-7 = 2008 Hz (sniff) / 2002 Hz (AdatNet capture) — see §0 |
| **Next action** | none — ADAT IN fixed and verified; see §0 for the two root causes |

---

## 2. Environment (do not change casually)

- **VM** (Debian 13, kernel `6.12.118+deb13-amd64`). The FPGA toolchain does
  **NOT** run here (OOM). Never synthesize in the VM; never run big sims.
- **Board**: ADATface, Cyclone IV `EP4CGX150`, LAN8720 RMII PHY, TOSLINK.
  FPGA config is **volatile (JTAG/SRAM)**; a power cycle leaves it unconfigured
  (UART silent). Reflash after any power cycle.
- **Host build**: `~/quartus_build_host.sh` (= `ssh -T rouven@10.0.2.2`,
  triggers the host build in `gateware/build/`). Do NOT ssh directly.
- **Scratch venv** (only needed to REGENERATE the LiteEth MAC Verilog, not for
  normal rebuilds): `/tmp/etheradat-scratch/liteenv` → after a reboot recreate
  the symlink: `ln -s /mnt/Audio/adat-usb2-audio-interface/etheradat-scratch-saved /tmp/etheradat-scratch`.
- **USB Ethernet NIC**: ASIX `AX88179A` (`0b95:1790`), MAC `f8:e4:3b:90:77:75`,
  ifname `enxf8e43b907775`, driver **`ax88179_178a`**. It MUST be on the
  **USB2 (480 M) bus** (`bus=xhci.0`). On USB3/SuperSpeed (5000 M) the driver
  fails PHY register reads (`reg 0x0040: -32`) → no link. `cdc_ncm` is
  blacklisted (`/etc/modprobe.d/90-ax88179-native.conf`) so the native driver
  is used from boot; a udev rule forces vendor config 1 if needed. The VM's
  primary NIC `enp0s2` is **virtio-net** and is unaffected.
- **FPGA UART**: FT232R → `/dev/ttyUSB0` (115200 8N1). FT232R is USB
  full-speed (12 M) by nature — fine, it carries no audio.
- **Audient iD14**: USB audio `2708:0002` at 480 M, ALSA `hw:0,0`, 10 capture
  channels (0-1 analog, 2-9 ADAT). Output path uses it.

---

## 3. Physical routing (as of this handoff)

- ADAT OUT 1-8  → Audient iD14 ADAT input (Audient capture ch2-9). **Verified.**
- ADAT OUT 9-16 → **ADAT IN 1-8** (self-loopback cable). (Changed by the user;
  it was OUT 9-16 → IN 9-16 earlier.) So the OUT2 tone should appear on
  **input0 / IN1**, i.e. adatnet capture ch0-7 and TX-frame slots `s*3+0`.
- The FPGA is the ADAT master (12.288 MHz audio clock). LAN8720 is
  `--ref-clk-mode input` (PHY provides the 50 MHz RMII REF_CLK).

---

## 4. Gateware architecture / where the data flows

Clock domains: `fast` (98.304 MHz, ADAT RX + builder), `sync` (61.44 MHz,
ADAT TX + feeder), `rmii` (50 MHz REF_CLK: bridge FSMs, MAC user streams,
discovery, MDIO), `sys` (50 MHz board osc: UART). Crossings are gray-coded
mailboxes / FFSynchronizer.

- **TX (device→PC)**: ADATReceiver ×3 → InputFrameCollector rings →
  `AudioBlockBuilder` (`etheradat/audio_payload.py`) → TX block RAM →
  `EtherDATBridge` TX FSM (`etheradat/bridge.py`) → LiteEth MAC user sink
  (`liteeth/liteeth_etheradat_mac.v`) → RMII serializer → LAN8720.
- **RX (PC→device)**: MAC user source → `EtherDATBridge` RX parser → RX block
  RAM → `AudioFeeder` + `FrameDeserializer` (`etheradat/adat_frame_codec.py`)
  → ADATTransmitter ×2 → TOSLINK out.
- The MAC blackbox `liteeth/liteeth_etheradat_mac.v` is generated by
  `liteeth/gen_etheradat_mac.py` (scratch venv). It is **committed with the
  bring-up taps** (see §6) — a normal rebuild uses it as-is.

---

## 5. Committed fixes (commit `038afcf`) — DO NOT re-break

1. **RMII TX clock edge.** The bridge used to drive the MAC `eth_tx` domain
   from an inverted `~REF_CLK`; Quartus does not route that as a clean clock →
   the serializer emitted a bit-shifted stream and the first ~18-20 bytes
   (Ethernet+AdatNet header) vanished. Fix: TX runs on the **rising REF_CLK
   edge** (`eth_tx_clk = rmii_clk`). `etheradat/scripts/generate_quartus_files_etheradat.py`
   now defaults to rising (`--tx-negedge` only for diagnosis).
2. **TX block-RAM payload read off by one byte.** The fit infers the 2880×8
   TX block as M9K with **input registers but no output register**, whose read
   timing differs by one cycle from the Amaranth model. Fix in `bridge.py`:
   `read_next = tx_byte - TX_HDR_BYTES` (address the *presentation* cycle),
   `en` from `tx_byte >= TX_HDR_BYTES`. Result: payload slots exact (§7).
   **Caveat:** this offset is tuned to this fit. The Amaranth **sim model is
   1 cycle different**, so the TX byte-exact unit test in
   `etheradat/tests/test_bridge.py` may disagree with hardware. Hardware is
   authoritative here. If a future fit adds an output register, flip the
   offset back to `tx_byte - (TX_HDR_BYTES-1)`.
3. **Unsynced inputs → exact `EMPTY_USER_FRAME`.** `audio_payload.py` writes
   the constant silence frame for any input with `synced_in == 0` (PROTOCOL
   §6), instead of trusting the collector ring.

---

## 6. Bring-up taps + UART (already in the build)

The top `etheradat/etheradat_interface.py` and the MAC blackbox expose debug
taps dumped over UART (`etheradat/bringup_tools/parse_eduart.py`):

- `sink_*` = bytes the bridge hands the MAC (first 16 + length/count).
- `mac_*`  = bytes the MAC hands the serializer (preamble+frame+FCS).
- `pin_*`  = reassembled RMII TX dibits (has a known width bug in the
  nibble reassembly; the *length* is still valid).
- `pay`    = first 16 payload bytes as the bridge hands them to the MAC.

Layout: `AA 01 <56×u16 LE> 5A` every 250 ms. `parse_eduart.py` decodes it.
`status0` = `{link, pc_linked, free_run, sync0, sync1, sync2, phy_found}`
(bit0..bit6). `status1` = `{feed_active, feed_commit, refclk_level, crs_dv}`.
`link`/`phy_found` are MDIO link/phy-found and are always 0 (MDIO bus is
dead on this board — ignore).

---

## 7. Evidence for "ADAT OUT works, ADAT IN broken"

TX payload sanity (`slot_by_input.py`), free-run, no input: 45/45 slots exact
`EMPTY_USER_FRAME`; driver decodes (link up, `eth_rx_frames` ~3200/s,
`eth_rx_errors` ~0.05 %).

Output (`loop_full.py`): `aplay tone16.wav` (ch0-7=1 kHz, ch8-15=2 kHz) →
**Audient capture ch2-9 = 993 Hz, rms ≈ 1.5e8** (clean). So ADAT OUT1 carries
the tone and the FPGA ADAT TX encoder is right.

Input: play `tone_out2.wav` (ch8-15 = 2 kHz → OUT2 → IN1):
- `status0` during play: **`sync0 = 1`** (IN1 clock-synced), `pc_linked = 1`.
- `underflow` **flat**, `feed`/`rx_frames` advancing → the feeder drives BOTH
  ADAT outputs; OUT2 is being transmitted (not starved).
- BUT decoding the FPGA TX frames for input0 (`loop_sniff_all.py`) gives
  mostly the **exact EMPTY frame** `00 21 84 21 08 42 … 10` (silence), with
  occasional garbage frames (FFT peaks 3200/22400 Hz, rms ~1.9e6 vs the
  ~1.5e8 a real 2 kHz tone would give). No clean 2 kHz.

Conclusion: the failure is in the **ADAT receive path** (ADATReceiver →
InputFrameCollector → builder), not the network/driver/output. The receiver
locks (`sync0=1`) but does not deliver correct samples.

(The PC/driver side is correct: `loop_sniff_tx.py` shows the PC playback
frames carry out0 ch0-7 = 1000 Hz and out1 ch0-7 = 2000 Hz.)

---

## 8. Build / flash / driver procedure (exact)

```bash
cd /mnt/Audio/adat-usb2-audio-interface/gateware

# 1) generate Quartus files (VM, project venv — NOT synthesis)
LUNA_PLATFORM=platforms:ADATFaceCycloneIV_EP4CGX150 \
  venv/bin/python etheradat/scripts/generate_quartus_files_etheradat.py \
  --ref-clk-mode input /tmp/rebuild
rm -rf build && mkdir build && cp /tmp/rebuild/* build/

# 2) synthesize on the HOST (never in the VM)
~/quartus_build_host.sh            # writes build/top.sof ; ~1-2 min
ls -la build/top.sof               # check it is fresh (stale-mount gotcha)

# 3) flash (volatile) — use a NEW filename each time (stale shared-mount)
cp build/top.sof build/fixN.sof
sudo pkill -9 jtagd; sudo /opt/altera_lite/25.1std/quartus/bin/jtagd
/opt/altera_lite/25.1std/quartus/bin/quartus_pgm -c 'USB-Blaster [1-1]' \
  -m JTAG -o 'P;build/fixN.sof'

# 4) (only after a kernel change) rebuild the driver for the running kernel
cd /mnt/Audio/etheradat/virtual_soundcard/kernel_module && make clean && make
sudo modprobe -r adatnet; sudo insmod ./adatnet.ko mode=1
sudo cat /proc/asound/card1/adatnet      # link up, hello_rx rising
```

Timing (last build): `rmii` setup slack ≈ +7 ns, `fastpll` ≈ +1.4 ns, all met.
The current flashed bitstream is `build/ufix3.sof` (checksum `0x0154EDEB`),
built from commit `038afcf`.

---

## 9. NEXT STEPS — fix the ADAT input

Goal: with `aplay tone_out2.wav` (2 kHz on OUT2 → IN1), decoding input0 from
the FPGA TX frames (`loop_sniff_all.py`) must show a clean ~2000 Hz on
IN1/input0 ch0-7, and the AdatNet capture must carry it.

Do it incrementally with **hardware + UART**, not sims (sims are slower here
and were misleading for the RAM-latency bug). Suggested order:

1. **Add an RX-side UART tap** (most informative, small change). In the top,
   expose for **input0**: `receiver.sample_out` (24b), `receiver.addr_out`
   (3b), `receiver.output_enable`, `receiver.synced_out`,
   `receiver.user_data_out`, and the collector's `dbg_frame_done`,
   `dbg_incomplete`, `slot_wrapped_out`. Latch the first N decoded samples of
   input0 (or a small snapshot per dump) into new UART values (extend
   `values` in `etheradat_interface.py` and `parse_eduart.py` NV/TOTAL).
   Run `aplay tone_out2.wav` and read the UART:
   - If `output_enable` never pulses / `addr_out` never 0..7 → the receiver
     isn't producing frames (despite `synced`): look at the receiver's bit
     clock / NRZI decode / enable logic.
   - If samples are delivered but wrong (garbage instead of a 2 kHz
     sinusoid) → the bit/time alignment is off (NRZI decode, tal-lock,
     wrap) or the wire↔payload bit convention differs by 1 bit.
2. Inspect `ADATReceiver` (from `adat` / the ADATFACE gateware) for the known
   prior RX issues: the project memory notes "NRZI wrap / dead-counter /
   tal-lock" fixes and "two ADAT bit conventions (wire vs payload bytes,
   shifted by 1 bit)". Compare the decoded samples of a **known** source:
   the FPGA's own OUT2 is a perfect source (it encodes the 2 kHz you played).
   Consider temporarily comparing the `ADATReceiver` output of the loopback
   against the simulator/RP2350 reference for the same bitstream.
3. Sanity gateways before deep FPGA work:
   - Confirm with `aplay tone_out2.wav` + `loop_full.py` that OUT2 is really
     the one looped to IN1 (swap to `tone_out1.wav` and expect IN1 silence if
     OUT1 goes to the Audient, not IN1).
   - Confirm the TOSLINK loopback signal is strong (if `sync0` only reaches 1
     marginally or flickers, the receiver can lock but mis-sample).
4. After a fix: `slot_by_input.py` (TX side unchanged), `loop_sniff_all.py`
   (input0 = 2 kHz), then `loop_full.py` / an `arecord` on `hw:1,0`
   (capture ch0-7 = 2 kHz), then `aplay`… full loop.
5. Possibly also verify the **reverse** ADAT-OUT-to-PC path is unaffected
   (OUT1→Audient still 1 kHz).

Likely root-cause candidates, in order of probability:
- ADATReceiver bit recovery (NRZI decode / tal-lock) mis-samples the looped
  signal even when `synced` (produces garbage or zeros).
- The collector only writes on a complete 8-channel frame
  (`enable_in & synced_in` + ch7 + seen 0..6); if `addr_in`/`output_enable`
  from the receiver are misaligned, the ring keeps its seeded `EMPTY` value →
  output is silence.
- Wire↔payload bit-convention offset (1 bit) between the receiver's decoded
  frame and what the builder/reference expects (this would give *scrambled*
  samples, which matches the occasional garbage).

---

## 10. Gotchas / failure modes

- **FPGA TX wedges on Ethernet link disturbance.** Whenever the link drops
  mid-stream (NIC re-enumeration, USB change, concurrent capture+playback on
  a bad link) the TX FSM can lock: `tx_frames`, `hello_tx`, `build` freeze and
  hellos stop. An ALSA concurrent `arecord`+`aplay` used to trigger it on
  half-duplex/12 M-USB. **Recovery = reflash the bitstream.**
- **USB speed matters.** NIC must be on the 480 M USB2 bus. USB3 (5000 M)
  breaks the AX88179A (`-32`). Do not move it to `xhci.1`.
- **`/tmp` is wiped by reboots** — recreate `/tmp/edtools`, the scratch
  symlink, and re-run `gen_tones.py` after each boot. That is why the tools
  now live in `etheradat/bringup_tools/` (repo).
- **`adatnet.ko` must match the running kernel.** A `.ko` built elsewhere
  (vermagic like `7.3.5-alderlake-xanmod`) fails `insmod`; rebuild in the VM
  with `make clean && make` (headers are in `/lib/modules/$(uname -r)/build`).
- **UART read can time out / come back empty** if `stty` was not run or the
  FTDI re-enumerated; re-run `sudo stty -F /dev/ttyUSB0 115200 cs8 -cstopb
  -parenb raw -echo`.
- **MDIO is dead** (`phy_found=0`, `link=0` in status0 are expected). The PHY
  runs on strap defaults; full-duplex is negotiated by the PC's native ASIX
  driver, not by the FPGA.

---

## 11. Repo files touched / relevant

- `etheradat/bridge.py` — TX read-address fix (§5.2), TX/RX bridge FSMs.
- `etheradat/audio_payload.py` — unsynced→EMPTY (§5.3), builder/feeder.
- `etheradat/adat_frame_codec.py` — `InputFrameCollector`, `FrameDeserializer`
  (RX store/read path — candidate for the input bug).
- `etheradat/etheradat_interface.py` — top, taps, UART dump.
- `etheradat/liteeth/gen_etheradat_mac.py` + `liteeth_etheradat_mac.v` —
  MAC blackbox with taps.
- `etheradat/scripts/generate_quartus_files_etheradat.py` — rising TX default.
- `etheradat/bringup_tools/` — the salvaged host tools (+ README).
- `ETHERADAT_STATUS_20261123.md` — full history.
