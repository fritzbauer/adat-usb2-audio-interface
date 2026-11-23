# EtherADAT-Interface — Implementierung & Stand 23.11.2026

## Ziel

Die vollständige EtherADAT-Brücke (AdatNet v1, Rolle 1 = RP2350-Seite)
auf der ADATface (EP4CGX150, LAN8720) zum Laufen bringen: 3 ADAT-
Eingänge und 2 ADAT-Ausgänge über Raw-Ethernet (EtherType 0x88B5/0x88B6),
Hello/Discovery, 3200 Blöcke/s, 15 Samples/Block, gemäß
`/mnt/Audio/etheradat/PROTOCOL.md`.

Aufbauend auf der verifizierten LiteEth-UDP/IP-Baseline vom 22.11.
(`ETHERADAT_STATUS_20261122.md`) wurde der MAC-Teil der Baseline
weiterverwendet (RMII-Shim, SFD-Realignment, SameClockCDC) und die
komplette Bridge (Discovery, AudioBlockBuilder, AudioFeeder, RX-Parser,
ADAT-Empfänger/-Sender) dahinter gehängt.

## Neue/geänderte Dateien

| Datei | Inhalt |
|-------|--------|
| `etheradat/liteeth/gen_etheradat_mac.py` | erzeugt `liteeth_etheradat_mac.v`: LiteEth-**MAC only** (User-Streams statt UDP/IP) mit denselben HW-Fixes wie `gen_baseline.py`, inkl. `SameClockRXCDC`. |
| `etheradat/liteeth/liteeth_etheradat_mac.v` | generierte Blackbox (Ports `user_sink_*`/`user_source_*`, alle Clocks = RMII-REF_CLK). |
| `etheradat/etheradat_interface.py` | neuer Top: Bridge + MAC-Blackbox + 3 ADAT-RX + 2 ADAT-TX + MDIO + UART-Dump. `--ref-clk-mode input` (Default) / `output`. |
| `etheradat/scripts/generate_quartus_files_etheradat.py` | Quartus-Dateien; SDC-Create-Clock für `rmii_0__ref_clk` **vor** `set_clock_groups` eingefügt. |
| `etheradat/audio_payload.py` | Free-Run-Takt korrigiert (siehe Bugs). |
| `etheradat/mdio.py` | MDIO-Preamble-Bug behoben (siehe Bugs). |
| `etheradat/clock_probe.py` + Skript | Bring-up-Diagnose: UART auf clk50, REF_CLK-Edge-Zähler, MDIO-Scan über alle 32 Adressen, rohe Pin-Pegel; optional mit PLLs. |
| `etheradat/adat_pattern_test.py` + Skript | „Hardcoded ADAT Data“: 1-kHz/2-kHz-Sinus auf OUT1/OUT2, OUT2→IN2-Loopback, UART-Dump von `synced2`/Sample. |
| `etheradat/uart_dump.py` | Referenz-UART-Dump (auf HW **nicht** verwendet, siehe unten). |

## Build & Flash

```bash
cd gateware
LUNA_PLATFORM=platforms:ADATFaceCycloneIV_EP4CGX150 \
  venv/bin/python etheradat/scripts/generate_quartus_files_etheradat.py \
  --ref-clk-mode input /tmp/edbuild       # oder output
rm -rf build && mkdir build && cp /tmp/edbuild/* build/
~/quartus_build_host.sh                    # Host-Synthese, NICHT direkt ssh
cp build/top.sof build/<neuer-name>.sof    # neuer Name gegen stale-Mount
sudo /opt/altera_lite/25.1std/quartus/bin/jtagd
/opt/altera_lite/25.1std/quartus/bin/quartus_pgm -c 'USB-Blaster [1-1]' \
  -m JTAG -o 'P;build/<neuer-name>.sof'
```

Timing (finaler Input-Build, 21 059 LEs / 14 %): `fastpll` 110.7 MHz
+1.47 ns, `sync` 61.44 MHz +7.95 ns, RMII +8.83 ns, clk50 +11.6 ns.
Ohne die SDC-Reihenfolge-Korrektur werden die RMII↔PLL-Pfade fälschlich
synchron bewertet (Setup-Verletzungen bis −8.9 ns).

## UART-Diagnose (115200, `/dev/ttyUSB0`)

Der UART läuft jetzt **immer** im `sys`-Domain-Takt vom 50-MHz-
Board-Oszillator (nicht mehr vom PHY-REF_CLK). Dump alle 250 ms:

```
AA 01 <14 × u16 LE> 5A
[0] txf   [1] rxf   [2] hello_tx  [3] hello_rx
[4] build [5] feed  [6] rxerr     [7] txdrop
[8] filt  [9] underflow [10] status0 [11] status1
[12] linked [13] hello_due-Pulse
```

`status0` = `{phy_found, synced[2:0], free_run, pc_linked, link}` (Bit 0 =
link). Lesen:

```bash
sudo stty -F /dev/ttyUSB0 115200 cs8 -cstopb -parenb raw -echo
timeout 3 sudo cat /dev/ttyUSB0 | xxd
```

## Auf Hardware verifiziert (23.11.2026)

- **Design läuft**: mit `--ref-clk-mode output` (FPGA treibt REF_CLK aus
  clk50) liefert der UART `build = txf = 3200/s`, `hello_tx = 4/s`,
  `txdrop` konstant. Die PLLs (60/12.288/110.592/61.44 MHz) laufen.
- **Free-Run-Rate exakt 3200/s** (800 Blöcke je 250-ms-Fenster) nach dem
  Timer-Fix.
- **ADAT-Empfänger + Sender**: `adat_pattern_test.py` (1-kHz-Sinus auf
  OUT1/OUT2) → OUT2→IN2-Loopback → `synced2 = 1` im UART. Damit sind
  Board, Audio-PLLs, TOSLINK-TX/RX und der ADAT-Codec auf Hardware
  bestätigt.
- **Voller Top**: `st0` zeigt wechselnd `free_run`/`synced1` (Eingang 2
  empfängt den eigenen OUT2-Loopback); `underflow` zählt 48 kHz
  (Sender wiederholen bei fehlenden PC-Daten das letzte Frame =
  ADAT-Nullsignal, wie gefordert).
- **Input-Modus mit totem REF_CLK** ist eindeutig diagnostizierbar:
  `build` bleibt bei 2 stehen (beide Puffer voll, rmii-Seite tot),
  `txdrop ≈ 3200/s`, `txf = hello_tx = 0`.

## Blocker: LAN8720 liefert keinen Takt und antwortet nicht auf MDIO

Der aktuelle Zustand der Hardware (23.11.):

- `clock_probe.py`: **REF_CLK-Edge-Zähler = 0** (Modul treibt keinen
  50-MHz-Takt), obwohl der Board-Oszillator (UART) läuft.
- MDIO-Scan über **alle 32 PHY-Adressen** mit korrigiertem 32-Bit-
  Preamble: keine Antwort (PHYID1 ≠ 0x0007), MDIO-Pegel beim Loslassen
  = 0 (kein externer Pull-up sichtbar).
- `--ref-clk-mode output` (FPGA treibt REF_CLK, Fallback für
  REF_CLK-IN-Strap) brachte ebenfalls keinen Link (PC-NIC: NO-CARRIER).

Interpretation: Das LAN8720-Modul ist **nicht versorgt oder nicht
angeschlossen** (J2:4/+3.3 V bzw. Jumper/Kabel). Der Board-Rest
(FPGA, Oszillator, PLLs, TOSLINK, UART) ist nachweislich in Ordnung.
Ohne Modul-Strom gibt es keinen REF_CLK, keinen Link und keine
Ethernet-Verifikation. Das ist kein Gateware-Fehler.

Zusätzlich: `Audient ADAT Clock Validity` blieb auch beim Sinus-Test
`off`; die iD14-ADAT-Eingänge sind laut User bekannt flaky. Der
OUT2→IN2-Loopback ist der verlässliche ADAT-Nachweis.

### Wenn das Modul wieder Strom hat

1. Board/Modul stromzyklieren (REF_CLK-OUT-Strap wird beim PHY-Power-On
   gesampelt), `--ref-clk-mode input` flashen.
2. UART: erwartet `txf = build = 3200/s`, `hello_tx = 4/s`,
   `st0` Bit0 (link) und Bit6 (phy_found) = 1.
3. PC: `192.168.42.1/24` auf `enxf8e43b907775`, `ping` irrelevant
   (kein IP im AdatNet), aber:
   `sudo insmod adatnet.ko mode=1` (aus `/mnt/Audio/etheradat/
   virtual_soundcard/kernel_module`), dann `cat /proc/asound/card0/adatnet`
   → `link`, `hello_rx` müssen steigen; `arecord` liefert 24 Kanäle.
4. Audio: `aplay` auf die 16 Playback-Kanäle → ADAT OUT1 (Audient 3–10),
   OUT2 → IN2-Loopback → Capture-Kanäle 9–16.

## Behobene Bugs (23.11.)

1. **Free-Run-Takt 1600 statt 3200 Blöcke/s** (`audio_payload.py`):
   Der 16-Bit-Timer wurde nur zurückgesetzt, wenn der Trigger exakt in
   einem `~copying`-Takt lag. Fiel der Periodenrand in einen Block-Copy
   (≈1800 Zyklen), lief der Timer bis 65536 → jeder zweite Block fehlte.
   Fix: Timer läuft kontinuierlich, setzt `free_tick`; der Trigger
   konsumiert den Tick (und löscht ihn auch beim Drop).
2. **MDIO-Preamble** (`mdio.py`): Das 46-Bit-Frame (32 Preamble + ST +
   OP + 10 Adr.) lag in einem 44-Bit-Register → nur 30 Preamble-Bits,
   Frame verschoben. LAN8720 antwortet darauf nicht. Fix: 46 Bit.
3. **SDC-Reihenfolge**: `create_clock rmii_0__ref_clk` wurde ans
   Dateiende angehängt, also **nach** `set_clock_groups`; die
   asynchrone Gruppe war leer → alle CDC-Pfade wurden synchron
   bewertet (bis −8.9 ns Setup). Fix: Constraint vor die Gruppen.
4. **MAC-Blackbox**: `gen_liteeth_core.py` (alt) hatte den
   SameClockCDC-Fix nicht; `liteeth_rmii_mac.v` war für die alten
   Port-Namen erzeugt. Neu: `gen_etheradat_mac.py` mit Fix +
   `liteeth_etheradat_mac.v`.
5. **Phantom** (dokumentiert, nicht verwendet): Der modulare
   `PeriodicDump`/`UartTX` (`uart_dump.py`) simuliert korrekt, zeigt
   auf Hardware aber **keine** Aktivität – in drei Builds (dbg-Domain
   = clk50, sync-Domain = 61.44 MHz, allein im ADAT-Test). Die
   identische Inline-FSM (Probe/Top) läuft. Deshalb nutzt der Top die
   Inline-Version; `uart_dump.py` bleibt als Referenz.

## Kernel-Treiber (23.11. neu gebaut und geladen)

- `linux-headers-6.12.118+deb13-amd64` nachinstalliert (die VM hatte nur
  .113/.116/.117; `KDIR` zeigte auf das fehlende .118).
- `make clean && make` im `kernel_module` → `adatnet.ko` mit
  `vermagic=6.12.118+deb13-amd64` (Rust-Codec-Objekt wird aus
  `rust_encoder/target/.../libadat_codec.a` neu erzeugt).
- `sudo /sbin/insmod adatnet.ko mode=1` → `adatnet0` als card 1,
  Discovery-Modus, `link down` (keine Hellos, weil die PHY-Strecke tot
  ist), `/proc/asound/card1/adatnet` lesbar.
- **Treiber selbst bestätigt**: `test_veth/test_discovery.sh 2` →
  „Discovery PASS (zero-config link + bit-exact audio + MAC rotation)“,
  96000 Frames × 24 Kanäle bit-exakt, link_ups=2, MAC-Rotation ok.
  Damit ist die PC-Seite (Treiber+Codec+Discovery) unabhängig verifiziert.

**Nicht bestätigt** ist nur die reale Ethernet-Strecke FPGA↔PC: dafür
muss das LAN8720-Modul wieder elektrisch leben (siehe Blocker).

## Letzte PHY-Rettungsversuche (23.11., ohne Erfolg)

- `J_2:22` (laut README NC, evtl. RST#) auf HIGH treiben: kein Effekt.
- `phy_reset_probe.py`: REF_CLK aus clk50 **treiben** + MDIO-Writes
  BMCR=0x8000 (Soft-Reset) an alle 32 PHY-Adressen, danach
  BMCR=0x3000 (Autoneg/100M) im Dauerlauf + nRST high:
  UART-Heartbeat zeigt hunderte gesendete MDIO-Frames, aber **kein
  Carrier** am PC-NIC und keine REF_CLK-Aktivität.
  → Das Modul ist elektrisch nicht versorgt/verbunden; per Software
  nicht rettbar.

Aktuell geflasht: **volles Interface im `--ref-clk-mode output`**
(FPGA treibt REF_CLK, `txf=build=3200/s`, Hello 4 Hz), Treiber geladen.
Sobald das Modul Strom bekommt, sollte der Link von selbst hochkommen;
sonst Modul stromzyklieren/reseaten (Strap wird beim PHY-Power-On
gesampelt) und ggf. `--ref-clk-mode input` flashen.

## Tests

- `python -m etheradat.tests.test_reference` (7 Tests) ✅
- `python -m etheradat.tests.test_discovery` (3) ✅
- `python -m etheradat.tests.test_adat_frame_codec` (3) ✅
- `python -m unittest etheradat.tests.test_bridge -v` (7 Tests, **OK**,
  1011 s) ✅
- `gen_etheradat_mac.py`-Regeneration: Scratch-venv
  `/tmp/etheradat-scratch/liteenv` (Symlink auf
  `etheradat-scratch-saved`).
