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

### Kontrollversuch: exakt die funktionierende Baseline

Der am 22.11. verifizierte Baseline-Bitstream wurde zu Sessionbeginn
gesichert (`/tmp/baseline_top.sof`, Checksumme `0x00A8666A`) und am
23.11. erneut geflasht; zusätzlich neu aus den Quellen gebaut
(`generate_quartus_files_baseline.py --ref-clk-mode input`) — **identische
Checksumme `0x00A8666A`**, aber:

- PC-NIC: `NO-CARRIER`, `ping 192.168.42.2` 0/3,
- UART: still (die Baseline wird vollständig vom REF_CLK getaktet).

Da exakt derselbe, am 22.11. funktionierende Bitstream heute keinen Link
mehr bekommt, liegt die Ursache **nicht** in der neuen Gateware, sondern
in der PHY-Versorgung/Verbindung. Der Zustand war schon beim ersten
Flash am 23.11. 07:21 so (vor allen Output-Mode-Experimenten).

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

## Nachtrag 2: Integrations-Debugging (nach Hardware-Power-Up)

Nach dem Power-Up des LAN8720 (der Blocker oben war echt: Modul ohne Strom)
wurde die Integration Schritt fuer Schritt debuggt. Verifizierte Meilensteine:

- **Baseline**: exakt der 22.11.-Bitstream (0x00A8666A) laeuft wieder,
  ping 5/5, UART-Zaehler steigen.
- **Kernel-Treiber** (neu gebaut, 6.12.118): veth-Discovery-Test PASS,
  24 Kanaele bit-exakt.
- **Hello-only MAC-Test** (statisches ROM): saubere 60-Byte-Hellos im
  tcpdump, Treiber `link up`, PC antwortet (role=2).
- **Discovery+MAC** (ohne Bridge/Audio): nach den Fixes saubere
  1462-Byte-Hellos, Treiber `link up` (`hello_rx` steigt).

Dabei gefundene und behobene Fehler:

1. **Discovery/MDIO im falschen Clock-Domain** (`bridge.py`): beide
   nutzen intern `m.d.sync`, liefen also im 61,44-MHz-`sync`-Domain der
   Audio-PLL, waehrend der TX-Arbiter in `rmii` (REF_CLK) laeuft. Der
   Hello-Handshake kreuzte Clock-Domains unsynchronisiert → Byte-Ausfaelle
   (fehlende Bytes, Fragmente). Fix: `DomainRenamer("rmii")` fuer beide.
2. **Hello-Daten-Mux**: `frame.word_select()` (272-Bit-Variablen-Shift)
   durch einen expliziten 34:1-`Array`-Mux ersetzt.
3. **RMII-TX-Pads registriert** (`rmii_shim.py`): kombinatorische Pads
   liessen die PHY am Byte-Lade-Rand gelegentlich den NAECHSTEN Dibit
   latchen (periodischer Bit-6-Fehler jedes 4. Byte, fit-abhaengig;
   mit gemischt lang/kurz nachgewiesen). `txd_r`/`tx_en_r` verzoegern den
   Strom um einen Takt.
4. **IFG**: `tx_gap_cycles = 200` am Shim (LiteEth-Gap zaehlt Takte, der
   Shim braucht 4 Takte/Byte; Default 12 = 3 Wire-Bytes statt 12).
5. **MAC-Konfiguration wie Baseline**: `with_store_and_forward=False`
   (Baseline hat keinen TX-Packet-FIFO; unser FIFO verhielt sich auf
   Hardware anders).
6. **Uniforme Hello-Laenge**: Der MAC/Padding-Pfad verschluckt kurze
   Frames in einem Strom aus 1462-Byte-Audioframes. Das Protokoll
   verlangt nur ≥20 Payload-Bytes → der Hello wird auf 1462 Byte
   aufgefuellt (wie ein Audioframe, kein Mixed-Length-Pfad).
7. **Atomarer Hello-Arbiter**: ohne `hello_active` konnte der Audio-FSM
   im selben Takt starten, in dem der Hello grantet wurde, und ihn im
   naechsten Takt preempten (ein Hello-Byte landet im Audioframe).
   Jetzt besitzt ein Hello den Pfad bis zu seinem `last`.
   Der Bridge-Sim liefert damit nur noch wohlgeformte 1462-Byte-Frames
   (72 Hellos + 18 Audio, 0 fehlerhaft).

**Verbleibender Fehler (offen):** Im vollen Bridge-Build sind die
Audioframes auf der Leitung noch korrupt: die vom PC empfangenen Frames
beginnen mit dem ADAT-Payload-Muster statt mit dst/src/EtherType — es
fehlen die **ersten 22 Bytes (der Ethernet+AdatNet-Header)**; die Laengen
sind 1443/1444 statt 1462. Der MAC-User-Stream der Bridge ist im
Amaranth-Sim byte-exakt, und der Discovery+MAC-Standalone-Build
uebertraegt auf derselben Hardware saubere 1462-Byte-Frames. Der
Unterschied ist der Bridge-Datenpfad (Header aus `hdr_byte`-Array +
Payload aus dem Block-RAM) bzw. der kontinuierliche 3200-fps-Strom.

**Naechster Schritt:** den Bridge-TX-Header auf Hardware sichtbar machen
(z. B. die ersten 22 User-Stream-Bytes in ein Debug-RAM schreiben und per
UART ausgeben) und mit dem ROM-Test vergleichen; alternativ den Header
testweise aus einem ROM statt aus dem Array speisen. Danach den
Audio-Payload-Pfad (BRAM-Read-Timing) pruefen.

## Nachtrag 3: Wire-Bug GELOEST (23.11.2026 abends)

Die beiden verbleibenden Fehler sind gefunden und behoben. Der
UART-Debug-Tap (erste 16 Header-Bytes bridge->MAC, MAC->Serializer und
Serializer->Pads, plus erster Payload-Block, `dbg_cap`/`dbg_pin`/
`sink_pay` in `etheradat_interface.py`) hat sie direkt auf Hardware
lokalisiert — ohne Sim.

### Fehler A: invertierter `~REF_CLK` als MAC-`eth_tx`-Clock

Der Top trieb `i_eth_tx_clk = ~rmii_clk` (`tx_negedge`, bisher Default).
Quartus leitet diesen invertierten Clock nicht sauber als Takt weiter;
der TX-Serializer gab dadurch einen **bit-verschobenen Strom** aus: die
ersten ~18-20 Byte jedes Frames fehlten (Header weg), der Payload erschien
verrauscht/verschoben. Der Tap zeigte: SINK (bridge->MAC) und MAC-Eingang
sind byte-korrekt (Header + `EMPTY`), erst am Serializer kam Muell heraus.
Der Standalone-Mixed-Test (gleicher MAC, **steigende** Flanke) war
sauber — der Unterschied war genau die Clock-Flanke.

**Fix:** TX laeuft jetzt auf der **steigenden** REF_CLK-Flanke (gleiche
Clock wie MAC). `generate_quartus_files_etheradat.py` nutzt Rising als
Default (`--tx-negedge` nur noch als Diagnose-Option).
Verifiziert: Frames kommen als korrekte 1462-Byte-`0x88b5`-Frames mit
Header und in-order `sample_cntr` (+15) an; PC empfaengt nur
`dir=0`-Frames, Treiber `link up`, `hello_bad 0`.

### Fehler B: Audio-Payload-Read um ein Byte verschoben

Selbst mit A kam der Payload um **ein Byte rotiert** an
(`EMPTY[1:]+EMPTY[:1]` pro Slot), der ADAT-Decoder des Treibers verwarf
jeden Frame (`decode_frame`: `frame[0] != 0` -> Err(Sync)) =>
`eth_rx_errors` ~ Frame-Rate, `eth_rx_frames = 0`. Ursache: der TX-Block-
RAM-Read in `bridge.py` adressierte eine Cycle zu weit vorne. Der
M9K-Inferenz im Fit hat **Input-Register, aber kein Output-Register**
(siehe `top.fit.rpt`: "Port B Input Registers yes / Output Registers no"),
d. h. das Read-Timing weicht um einen Takt vom Amaranth-Modell ab (Sim
war byte-exakt). Die alte Adresse `(tx_byte-TX_HDR_BYTES)+1` unterlief
bei `tx_byte=21` zusaetzlich unsigned.

**Fix:** Read-Adresse direkt fuer die **Praesentations-Cycle** berechnen:
`read_next = tx_byte - TX_HDR_BYTES`, `en` ab `tx_byte >= TX_HDR_BYTES`.
Damit liefert der Fit `block[tx_byte-22]`. Verifiziert: alle 45 Slots
byte-exakt `EMPTY_USER_FRAME` (0/45 -> 45/45), Treiber
`eth_rx_frames` ~3200/s, `eth_rx_errors` ~0.05 % (nur Host-Drops),
`frames_copied` steigt bei offenem `arecord`.

Zusaetzlich: `AudioBlockBuilder` schreibt fuer **unsynchrone** Eingaenge
jetzt explizit das exakte `EMPTY_USER_FRAME` in den Block (statt sich auf
den Ring-Init zu verlassen); das haelt PROTOCOL §6 ein, selbst wenn ein
verrauschter Receiver-Eingang den Ring einmal mit einem rotierten Frame
beschreibt.

### Verifizierter Zustand (Hardware)

- `link up`, `hello_rx` steigt, `hello_bad 0`, `pc_linked = 1`.
- TX: 1462-Byte-`0x88b5`-Frames, Header korrekt, Payload = exaktes
  `EMPTY_USER_FRAME` (Free-Run), `sample_cntr` +15; 34-Byte-`0x88b6`-
  Hellos.
- Treiber (mode=1): `eth_rx_frames` ~3200/s, `eth_rx_errors` ~0.05 %.
- `arecord -D hw:1,0 -f S32_LE -r 48000 -c 24` liefert 24 Kanaele; Inhalt
  im Free-Run korrekt still (0 Nicht-Null-Samples); `frames_copied`
  steigt. PC->ADAT: `eth_tx_frames`/FPGA-`rx_frames`/`feed` steigen.

### Offen / Hinweis

- Der Payload-Read ist auf das **Fit-Read-Timing** (M9K Input-Reg, kein
  Output-Reg) abgestimmt; das Amaranth-Sim-Modell ist 1 Cycle spaeter.
  Bei einem kuenftigen Fit mit Output-Register muss der Offset
  (`TX_HDR_BYTES`) entsprechend gedreht werden — sauber waere, das
  Read-Timing per Attribut festzunageln. Auf Hardware ist der aktuelle
  Stand verifiziert.
- `phy_found = 0` (MDIO antwortet nicht) und Link **100 Mbit Half/AN off**
  bleiben: vermutlich MDIO-Verdrahtung/Strap, beeinflusst den Datenpfad
  aber nicht. Der frueher vermutete Leitungs-/Noise-Fehler ist
  ausgeschlossen (0 CRC-/Length-Fehler; der Mixed-Rom-Test ist ueber
  dieselbe Verkabelung sauber).
- Debug-Taps (`dbg_cap`, `dbg_pin`, `sink_cap`, `sink_pay`) sind noch im
  Top (Bring-up); sie kosten ~600 FFs und vergroessern den UART-Dump.
  Fuer einen finalen Build koennen sie entfernt werden.

## Nachtrag 4: ADAT-INPUT GELOEST (23.11.2026 abends)

Zwei unabhaengige Ursachen (Details in
`ETHERADAT_HANDOFF_INPUT_20261123.md` §0):

1. **Fast-Clock 9x statt 8x.** `platforms.py` setzte fuer die Cyclone-IV-
   Klassen `fast_multiplier = 9`; `car.py` gibt das direkt als
   `CLK0_MULTIPLY_BY` an die Fast-PLL, real also 110.61 MHz statt
   98.304 MHz (im `top.fit.rpt` verifiziert). `NRZIDecoder` hardcodet
   `bit_time = 8`, d. h. der Bit-Zaehler lief 12.5 % zu schnell ->
   EMPTY/Garbage trotz `synced`. Fix: `fast_multiplier = 8`
   (Cyclone IV, EP4CGX150, Cyclone10) + Regressionstest
   `etheradat/tests/test_platform_clocks.py`.
2. **Builder-Trigger feuerte nie** (sobald `sync0` stabil war, `build = 0`
   bei Ring-Wrap 3200/s). `Array([wrap0..2])[master]` wird von Amaranth zu
   `casez(master)` mit Wildcard-Default; auf Cyclone IV / Quartus 25.1
   wurde Element 0 bei `master == 0` nicht selektiert. Fix: explizite
   Prioritaets-AND/OR (`master_wrap`) und explizite `Mux` fuer die
   Synced-Pfad-Byte-Auswahl.

Hardware-Verifikation (Bitstream `0x01503569`):
`tx_frames == build ≈ 3200/s`, `sync0 = 1`, `c_rx_fdone ≈ 48000/s`,
`c_rx_wrap ≈ 3200/s`, `c_rx_finc = 0`; `loop_sniff_all` (OUT2 2 kHz ->
IN1) input0 = 2008 Hz; `loop_full` AdatNet ch0-7 = 2002 Hz und Audient
ch2-9 = 1002 Hz. Der bisherige Ein-Byte-Sim-vs-Fit-Versatz in den
`test_bridge.py`-Payload-Asserts bleibt (dokumentiert, hardware-autoritativ).
