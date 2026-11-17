# EtherADAT-Bridge (ADATface)

Brückt 3 ADAT-Eingänge und 2 ADAT-Ausgänge der ADATface auf raw-Ethernet
(AdatNet-Protokoll, Rolle 1 = die „RP2350-Seite“) über einen LAN8720-PHY
an RMII. Das ist die FPGA-Umsetzung des gleichnamigen RP2350-Firmware-
Teils aus `/mnt/Audio/etheradat` (PROTOCOL.md, EtherType 0x88B5 Audio /
0x88B6 Hello, kein IP/UDP/TCP).

Die PC-Seite (Kernel-Modul `adatnet.ko` in `/mnt/Audio/etheradat/
virtual_soundcard/`) funktioniert unverändert gegen diesen Bridge.

## Aufbau

```
toslink 1..3 (RX-Optik)  →  ADATReceiver ×3  ─┐        „fast“ (98,304 MHz)
                                              │  AudioBlockBuilder
                                              │  (15-Sample-Ringe, Blockbau,
                                              │   Slot (s*3+i), EMPTY bei
                                              │   unsynchronisiertem Eingang,
                                              │   Free-Run 3200/s)
                                              ▼
                            RMII-Domäne (50 MHz): Ethernet-Frames
                            (Header/MAC-Lernen/Hello/Arbitrierung)
                                              │
                            LiteEth-MAC-Core (Verilog-Blackbox,
                            liteeth_rmii_mac.v: Preamble, FCS-CRC32,
                            Padding auf 60 B, IFG)
                                              │
                            RMII-Pins → LAN8720 → CAT-Kabel
                            
toslink 1..2 (TX-Optik)  ←  ADATTransmitter ×2 ← AudioFeeder ← RX-Frames
                             „sync“ (61,44 MHz)   (Slot (s*2+o), Repeat bei
                                                   Unterlauf, Drop bei
                                                   Überlauf)
```

Alle Puffer-Handovers zwischen den Taktbereichen laufen über
Gray-kodierte Zähler (2-bit, 2-Puffer-Ping-Pong) und FFSynchronizer.

## Verdrahtung J7 ↔ LAN8720-Modul

J7 liegt auf dem QMTech-J2 (gerade Pins, nebeneinanderliegend); das
Modul hat 12 Pins (2 Versorgung, 10 Signale):

| QMTech J2 Pin | Signal   | LAN8720-Modul          |
|---------------|----------|------------------------|
| 4             | +3.3 V   | VCC                    |
| 6             | GND      | GND                    |
| 8             | TXD0     | TXD0                   |
| 10            | TXD1     | TXD1                   |
| 12            | TX_EN    | TX_EN                  |
| 14            | RXD0     | RXD0                   |
| 16            | RXD1     | RXD1                   |
| 18            | CRS_DV   | CRS_DV                 |
| 20            | REF_CLK  | REF_CLK (Richtung s. u.) |
| 22            | nRST     | nRST                   |
| 24            | MDC      | MDC                    |
| 26            | MDIO     | MDIO                   |

(Hinweis: Die Pins 8/10/12 kollidieren mit der „spi“-Debug-Ressource
der Baseboard-Definition; beide Ressourcen dürfen nicht gleichzeitig
angefordert werden. Der EtherADAT-Top fordert RMII an, das
USB-Design benutzt die Debug-SPI nur mit aktiviertem ILA.)

### REF_CLK-Richtung (wichtig — Datenblatt prüfen!)

Der Top hat einen Parameter `ref_clk_mode`:

- **`"output"` (Default)**: Die FPGA erzeugt 50 MHz aus dem
  50-MHz-Oszillator des QMTech-Boards und **treibt** den Takt in das
  Modul. Dafür muss der LAN8720 in den **REF_CLK-In-Modus** gestrappt
  werden: laut Datenblatt (DS00002157B) wird der 50-MHz-Referenztakt
  in diesem Modus auf **XTAL1/CLKIN** gelegt und die Auswahl erfolgt
  über den **nINTSEL-Strap** (Pin 2, LED2/nINTSEL). **Prüfen, wohin
  der „REF_CLK“-Pin des konkreten Moduls geführt ist** (Pin 14 =
  nINT/REFCLKO oder XTAL1) und entsprechend verdrahten/strappen.
- **`"input"`**: Das Modul erzeugt den Takt selbst (25-MHz-Quarz,
  REF_CLK-OUT-Strap, der bei den meisten fertigen Modulen Default
  ist) und die FPGA takten ihre RMII-Domäne vom REF_CLK-Eingang.

In beiden Fällen wird der Modul-nRST von der FPGA getrieben (Reset
bis nach dem PLL-Lock).

## MDIO

MDC/MDIO sind angeschlossen, aber **read-only**: Der Bridge tastet
per PHY-Identifier (Register 2 = 0x0007) die PHY-Adresse ab (Straps
0 oder 1) und pollt das Link-Bit (BMSR, Register 1, Bit 2) für die
Link-LED. Konfiguriert wird der PHY nicht — er läuft auf seinen
Strap-Defaults (Auto-Negotiation, 100 Mbit/s fullduplex gegen den PC).

## LEDs (Bring-up)

| LED      | Bedeutung                              |
|----------|----------------------------------------|
| host     | PHY-Link (BMSR)                        |
| usb1     | PC-MAC gelernt (Discovery gelinkt)     |
| usb2     | Free-Run (kein ADAT-Eingang sync)      |
| sync1..3 | ADAT-Eingang 1..3 synchronisiert       |
| active1  | Aktivität TX-Frames (MSB des Zählers)  |
| active2  | Aktivität RX-Frames                    |

## MAC/Identität

- `own_mac`: Default `02:AD:6E:65:74:01` (lokal verwaltet),
  Parameter des Tops.
- `device_serial`/`session_nonce`: Hello-Felder; der Nonce ändert sich
  pro Boot (am Reset gezogen), damit der PC den Reboot erkennt.

## Tests

Im `gateware/`-Verzeichnis mit aktiviertem Projekt-venv:

```
python -m etheradat.tests.test_reference          # Referenzmodell
python -m etheradat.tests.test_adat_frame_codec   # 32-Byte-Codec + Draht-Roundtrip
python -m etheradat.tests.test_discovery          # Hello/MAC-Lernen/Keepalive
python -m unittest etheradat.tests.test_bridge -v # End-to-End an der MAC-Grenze
```

Zusätzlich (Separates venv mit migen/litex/liteeth, siehe
`liteeth/gen_liteeth_core.py`): der RMII-Loopback des LiteEth-MAC-Cores:

```
/tmp/etheradat-scratch/liteenv/bin/python etheradat/liteeth/test_rmii_shim.py
```

## Erzeugen der LiteEth-Blackbox (nur bei Bedarf neu)

Die Datei `liteeth/liteeth_rmii_mac.v` ist eingecheckt; neu generieren:

```
python3 -m venv /tmp/etheradat-scratch/liteenv
/tmp/etheradat-scratch/liteenv/bin/pip install migen liteeth
/tmp/etheradat-scratch/liteenv/bin/python etheradat/liteeth/gen_liteeth_core.py
```

## Build

Wie gehabt über den LUNA-Flow, Default-Plattform Cyclone V:

```
cd gateware && . venv/bin/activate
python etheradat/etheradat_interface.py            # oder:
LUNA_PLATFORM=platforms:ADATFaceCycloneIV python etheradat/etheradat_interface.py
```

Die Quartus-Projektdateien inkl. der Blackbox entstehen wie bei
`scripts/generate_quartus_files.py` (die Blackbox wird per
`platform.add_file` in den Build-Plan aufgenommen).

## Bewusste Einschränkungen (v1)

- **Kein Retransmit/NAK** (PROTOCOL.md §12.5) — Frames werden
  verworfen, der PC-Adapter füllt Lücken. Retx-Hooks sind später
  einfach ergänzt (Hello-Flag bit2, NAK-Frame = Hello-Variante).
- **Nur Mode A** (raw 32-Byte-ADAT-Frames als Payload). Mode B
  (Onboard-Codec) ist für die FPGA nicht sinnvoll (es dekodiert
  nativ) und wird nicht unterstützt; der Bridge sendet `ver_flags`
  mit Mode-Bit 0.
- **48 kHz only** (wie das USB-Design; das Audiopll fest auf
  12,288 MHz).
- Die 2 ADAT-Ausgänge laufen auf dem lokalen 12,288-MHz-Takt; die
  PC-Seite richtet ihre Senderate nach dem `sample_cntr` der
  TX-Frames aus (PROTOCOL.md §6.2), Restdrift übernimmt der
  Pufferadapter (Repeat/Drop).
