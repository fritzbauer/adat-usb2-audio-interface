# Firmware-Testbericht ADATface (Cyclone V 5CEFA2) — 13.11.2026

## Ziel
Fortsetzung des Berichts `firmware_test_report_2026-11-13.md` (Cyclone IV EP4CGX150), gleiche
Prozedur für das QMTech-Cyclone-V-Board (5CEFA2F23C8): Alle vorhandenen Cyclone-V-`.sof`-Dateien
**nur volatil per JTAG** testen und dokumentieren, welche brauchbar sind; danach den Flash
**nur lesend** auslesen und den Inhalt identifizieren. Der Flash wurde zu keinem Zeitpunkt
beschrieben — ausschließlich `quartus_pgm -m jtag -o "p;…"` (volatil) und
`openFPGALoader --dump-flash` (read-only) wurden benutzt.

## Setup
- Quartus Lite 25.1std: `/opt/altera_lite/25.1std/quartus/bin/`
- USB-Blaster `09fb:6001`, Schreibrechte per `sudo chmod a+rw /dev/bus/usb/001/<nr>`
- Board: QMTech 5CEFA2F23C8, JTAG-ID `0x02B150DD` (`jtagconfig` meldet `5CE(BA2|FA2)`)
- USB-Audiogerät `1209:ada1` „ADATface (USB1)", OpenAudioGear, bcdDevice 0.01
- Flash-Chip: **Micron N25Q128** (JEDEC `0x20BA18`, 128 Mbit = 16 MiB)

## Versionen — alle Cyclone-V-`.sof` (chronologisch, 12 Stück)

| Datei | Build-Datum | git-Kontext (aus Commit-Daten) | Test |
|---|---|---|---|
| adatface-5CEFA2-v0.0.8.sof | 2022-01-25 | Commit `c50c7a6` „Cyclone V PLL is working now" (25.01.) | ✅ PASS |
| build_orig/top.sof | 2022-02-12 | zwischen `c50c7a6` (25.01.) und `e0baf48` „first non-working version of fir experiments" (02.02.) | ✅ PASS |
| top_fir4096.sof | 2022-03-09 | `15aafca` „it works with 4096!" (09.03.) | ✅ PASS |
| adatface_fir.sof | 2022-04-01 | `522d21a`/`c57c292` „refactoring" (01.04.) | ✅ PASS |
| adatface_fir2.sof | 2022-04-04 | zwischen `522d21a` (01.04.) und `04fc7dc` (07.04.) | ✅ PASS |
| adatface_fir3.sof | 2022-04-08 | `e645b49` „refactoring and documentation" (08.04.) | ✅ PASS |
| adatface_fir4.sof | 2022-04-08 | `bef7593` „updated IR file" (08.04.) | ✅ PASS |
| adatface_fir5.sof | 2022-05-20 | `eb93436` „added new IR file with delay of 0,5ms between channels" (20.05.) | ✅ PASS |
| adatface_fir6.sof | 2022-06-01 | kein Commit am 01.06. — zwischen `eb93436` (20.05.) und `6103c7d` (03.06.), uncommitted | ✅ PASS |
| ../build/top.sof (Repo-Root) | 2022-06-25 | nach `1dc04d6`/`0e204ab` „4 FFTs … consumes too much resources" (06.06.), uncommitted | ✅ PASS |
| top_litex_I2s.sof | 2022-11-27 | keine Commits 06.2022–02.2024 in diesem Bereich | ✅ PASS |
| build_i2s/top.sof | 2022-12-06 | LiteX-SoC-Build (`VexRiscv.v`, `qmtech_5cefa2.v`), uncommitted | ✅ PASS |

Nicht testbar am Cyclone-V-Board (anderes FPGA, EP4CGX150): `adatface_EP4CGX150.sof`,
`adatface_EP4CGX150_32chan_longIR.sof`, `build_quartus23.1/top.sof`, `build/top.sof` (2024) —
diese gehören zum Cyclone-IV-Bericht.

## Testprozedur
1. `quartus_pgm -m jtag -o "p;<sof>"` (volatil, ~20 s)
2. 5 s warten; das Gateware disconnectet sich danach selbstständig (USB-Devnum wechselt),
   zusätzlich frische Enumeration erzwungen per USBDEVFS_RESET
   (`tools/firmware_test/usb_reset.py`, mit sudo)
3. 20 s warten; prüfen: `lsusb`, `/proc/asound/cards`, `aplay -l`, `arecord -l`, `sudo dmesg -T`
4. Stabilität: `arecord -D hw:0,0 -c 2 -r 48000 -f S32_LE -d 2 -t raw /dev/null`
   (Kanalzahlen 2/8/36/16/4 der Reihe nach, bis eine öffnet) + 5 s später Geräte-Präsenz-Check
- Skript: `tools/firmware_test/test_one.sh <label> <sof>`; Logs in `firmware_test_logs_cycloneV/`

## Ergebnisse
**Alle 12 Versionen PASS.** In allen 12 Logs:
- ALSA-Karte `ADATfaceUSB1` registriert, **Playback + Capture** vorhanden (je 1 PCM-Device)
- 2ch-Capture-Stream (48 kHz, S32_LE) öffnet und läuft 2 s fehlerfrei
- Gerät bleibt danach vorhanden (kein Disconnect, kein „DISAPPEARED")
- JTAG-Load → USB-Disconnect → selbstständige Re-Enumeration (bei allen Versionen)

Einschränkung (bewusst, nicht bewertet): getestet wurde nur die **USB/ALSA-Ebene** —
Enumeration, PCM-Open und fehlerfreier Stream. **Nicht** geprüft: Audio-Inhalt
(Verzerrungen, ADAT-Laufwerk/Loopback, Crossfeed/FIR-Funktion, MIDI, ILA-Interface).

### dmesg-Auffälligkeiten (wie beim Cyclone IV)
- `config 1 interface 3 altsetting 0 bulk endpoint 0x3/0x83 has invalid maxpacket 64`
  (bei jeder Enumeration; harmlos, Diagnose-/ILA-Interface)
- Während der Selbst-Enumeration direkt nach JTAG-Load treten transient
  `device descriptor read/64, error -110` auf (Boot-Phase des Softcore-USB); danach
  enumiert jede Version sauber. Die `parse_audio_format_rates_v2v3()`-Fehler der
  Cyclone-IV-Versionen (01.02.–09.02.2024) traten hier **nicht** auf.

## Flash-Inhalt — identifiziert (read-only)
- Dump (16 MiB): `flash_backup_5CEFA2/flash_backup_5CEFA2_2026-11-13.bin`
  - md5 roh: `7e2b28b32b1ff0a8def2c701305fb22a`
  - md5 bit-reversed (pro Byte LSB/MSB getauscht): `a111528df349396395784fdf5e4a0a74`
- Lesen mit `openFPGALoader --cable usb-blaster --fpga-part 5CEFA2F23C8 --dump-flash
  --file-size 16777216 -o 0 …` (dauert ~10 min; lädt dafür die Bridge volatile ins FPGA)
- **Identifikation:** Der Flash-Datenbereich (Offset 128 … 0x2829CF, 2 632 144 B ≈ 2,51 MiB)
  ist **bit-identisch** (0 Diffs) mit der Datenpage von `adatface_fir6.jic` ab Datei-Offset
  `0x11B` (Byte-reversed verglichen). Alles dahinter: 0xFF (gelöscht).
- Zusatzcheck gegen sämtliche andernorts vorhandenen Images: alle 12 `.sof`→`.rbf`
  (quartus_cpf) verglichen — nur `adatface_fir6.rbf` deckt sich zu 100 % (8 unbedeutende
  Header-Byte-Unterschiede < 0xA8 durch andere Konvertierungsoptionen der Original-JIC);
  alle übrigen ~50–60 %. Die 5CEFA2-`.jic`s fir…fir5, top_fir4096, v0.0.8 und die
  Cyclone-IV-`.jic`s passen nicht.
- ⇒ **Im Flash läuft adatface_fir6 (Build 2022-06-01)**, programmiert aus
  `adatface_fir6.jic`. Diese Version ist zugleich einer der bestandenen Testfälle.
- Bridge-Bitstream: `spiOverJtag_5CEFA2F23C8.rbf.gz` fehlt im openFPGALoader-Paket;
  Abhilfe: Symlink auf das beiliegende `spiOverJtag_5ce223.rbf.gz` (gleiche
  5CE2/F23-Klasse):
  ```
  sudo ln -sf /usr/share/openFPGALoader/spiOverJtag_5ce223.rbf.gz \
              /usr/share/openFPGALoader/spiOverJtag_5CEFA2F23C8.rbf.gz
  ```
  (kontrolliert: `openFPGALoader --detect` → 5CE*A2, JEDEC-Read OK, Dump ok)
- Nach dem Dump läuft die Bridge im FPGA (USB-Audio weg). Zurückgeladen per JTAG
  (`adatface_fir6.sof`, volatil) — alternativ RST-Knopf (lädt Flash-Version).

## Timing-/Resource-Befunde (Cyclone-V-Build-Verzeichnisse)
| Build | ALMs | DSP | RAM-Blöcke | Timing (Slow 1100mV 85C Setup) |
|---|---|---|---|---|
| build_orig (2022-02-12) | 4 770 (51 %) | 0 | 98/176 | **audiopll general[2]: Slack −6.476** (TNS −177.267); mainpll +2.004; audiopll[3] +2.197 |
| ../build (2022-06-25) | 6 030 (64 %) | 0 | 74/176 | alle Slacks positiv: mainpll +4.181, audiopll[2] +3.761, audiopll[3] +2.058 |
| build_i2s (2022-12-06) | 7 774 (82 %) | 4 | 117/176 | **litex_soc PLL −14.498, audiopll[2] −6.387, mainpll[0] −5.632** (TNS −650.796) |

- Die Verletzungen liegen auf internen PLL-Zählerpfaden (`PLL_OUTPUT_COUNTER|divclk`).
  Ob sie sich funktional auswirken, wurde nicht geprüft (kein Behauptung, nur Befund).
  Fakt: `audiopll[2]` ist in build_orig (02.02.) verletzt, in ../build (25.06.) intakt —
  dazwischen wurde am audiopll etwas geändert.
- Die `top.fit_*.rpt` im Repo-Root (12200MAC, 12288FFT, 2048FFT, 33210MAC, 34440MAC,
  4096FFT — alle Juni 2022) sind **Cyclone-IV-EP4CGX150**-Experimente (FFT- vs. MAC-
  Ressourcenvergleiche), nicht Cyclone V.

## Zuordnung / woran es liegen könnte
- Anders als beim Cyclone IV gibt es hier **kein Fehlerbild zu erklären**: alle Builds
  bis 2022-12-06 enumieren und streamen fehlerfrei. Das im Flash stehende fir6 (2022-06-01)
  war zum Zeitpunkt der Tests sogar die Flash-Referenz.
- Auffällig aus dem git log (Fakten, keine Bewertung): der FFT-Convolver-Phase 02.–06.06.2022
  folgend wird im Repo-Root-`build` (25.06.) mit Stand nach „4 FFTs … consumes too much
  resources" gebaut — hier ist das Timing (im Gegensatz zu build_orig/build_i2s) sauber.
  Die Phase nach fir6 (06.02.–06.06.) enthält Commits wie „it only outputs silence so far"
  (`69924d3`) — fir6 selbst (01.06., uncommitted) verhält sich am Gerät fehlerfrei.
- Womit **nicht** getestet wurde: Audioqualität der FIR-Convolver-Versionen. Alle
  „brauchbar"-Aussagen dieses Berichts beziehen sich auf die USB/ALSA-Ebene.

## Rezepte (für Wiederholung)
```bash
# volatile JTAG-Load (niemals Flash beschreiben!)
quartus_pgm -m jtag -o "p;<sof>"
# Flash nur LESEN (16 MiB, ~10 min, Bridge wird volatile geladen)
sudo openFPGALoader --cable usb-blaster --fpga-part 5CEFA2F23C8 \
  --dump-flash --file-size 16777216 -o 0 dump.bin
# Dump ↔ JIC-Vergleich: Dump pro Byte bit-reversen, dann mit jic[0x11B:] vergleichen
```
RBF aus SOF extrahieren (geht auch ohne Cyclone-V-Device-DB):
```
quartus_cpf -c --option=bitstream_compression=off <sof> <rbf>
```

## Testartefakte
- Logs (12 Tests): `gateware/firmware_test_logs_cycloneV/*.log`
- Flash-Dump: `gateware/flash_backup_5CEFA2/flash_backup_5CEFA2_2026-11-13.bin`
- Extrahierte RBFs: `gateware/flash_backup_5CEFA2/rbf_extract/`
- Testskripte: `gateware/tools/firmware_test/{test_one.sh,usb_reset.py}`
