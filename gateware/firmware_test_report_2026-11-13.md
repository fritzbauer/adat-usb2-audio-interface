# Firmware-Testbericht ADATface (Cyclone IV EP4CGX150) — 13.11.2026

## Ziel
Testen, ob die gebauten Firmware-Versionen sich ordnungsgemäß als ALSA-Audiogerät melden,
ohne die funktionierende Version im Flash zu überschreiben. Alle Tests wurden **nur volatil
per JTAG** durchgeführt (`quartus_pgm -m jtag -o "p;file.sof"`), der Flash-Inhalt wurde
nirgends beschrieben.

## Setup
- Quartus Lite 25.1std: `/opt/altera_lite/25.1std/quartus/bin/`
- USB-Blaster `09fb:6001` — für `jtagd` Schreibrechte nötig:
  `sudo chmod a+rw /dev/bus/usb/001/<nr>` (Nummer aus `lsusb`, ändert sich nach USB-Reset!)
  (dauerhaft besser: udev-Regel für 09fb:6001, 1209:ada1)
- Board: QMTech EP4CGX150DF27I7, JTAG-ID `0x028040DD`
- Gerät über USB-Passthrough der VM (udev-Regel des Hosts für automatische Wiederverbindung beachten)

## Versionen — Zuordnung FPGA-Typ
Alle `.sof`-Dateien tragen die Device-ID im Klartext (lesbar mit `strings -e l` bzw. `strings`).
Die `.map`-Dateien der `.jic`s geben das Konfigurationsdevice der Konvertierung an.

| Datei | FPGA | Testbar am EP4CGX150-Board? |
|---|---|---|
| adatface_EP4CGX150.sof (2024-02-01) | EP4CGX150DF27I7 | **ja** |
| adatface_EP4CGX150_32chan_longIR.sof (2024-02-04) | EP4CGX150DF27I7 | **ja** |
| build_quartus23.1/top.sof (2024-02-08) | EP4CGX150DF27I7 | **ja** |
| build/top.sof (2024-02-09, neueste) | EP4CGX150DF27I7 | **ja** |
| adatface-5CEFA2-v0.0.8.sof | 5CEFA2F23C8 (Cyclone V) | nein (anderes Board) |
| adatface_fir.sof, adatface_fir2–fir6.sof | 5CEFA2F23C8 | nein |
| top_fir4096.sof, top_litex_I2s.sof | 5CEFA2F23C8 | nein |
| build_orig/top.sof, build_i2s/top.sof, ../build/top.sof | 5CEFA2 (qmtech_5cefa2) | nein |

Hinweis: `build_quartus23.1/top.sof` ist trotz Verzeichnisnamen ein Cyclone-IV-GX-Build;
die 2,4-MB-`top.sof` in `build_orig/`, `build_i2s/` und im Repo-Root `build/` sind
Cyclone V (gleiche Größe/md5-Klasse wie `adatface-5CEFA2-v0.0.8.sof`).

Die `.jic`s `adatface-fir`, `adatface_fir2–fir6`, `adatface-5CEFA2-v0.0.8` und `top_fir4096`
sind laut `.map` für 5CEFA2 konvertiert, nur `adatface_EP4CGX150*.jic` für EP4CGX150.

## Testprozedur (je Version)
1. `quartus_pgm -m jtag -o "p;<sof>"` (volatil, ~40 s)
2. Frische USB-Enumeration erzwingen:
   - Oft enumiert das Gerät nach dem JTAG-Load selbst neu (Device-Nummer wechselt).
   - Sonst Bus-Reset vom Host senden (siehe unten).
3. 20 s warten, dann prüfen: `lsusb | grep ada1`, `/proc/asound/cards`, `aplay -l`, `arecord -l`, `dmesg -T`
4. Stabilitätstest: `arecord -D hw:0,0 -c 36 -r 48000 -f S32_LE -d 2 -t raw /dev/null`
   → danach 5 s warten, Gerät muss weiterhin da sein.

### Wichtig gelernt: Reset-Verhalten des Boards
- **USB-Kabel neu einstecken = Power-Cycle** (Board hängt am USB-Bus) → lädt die **Flash-Version**
- **RST-Knopf** → rekonfiguriert aus dem Flash (nCONFIG-Verdrahtung, empirisch verifiziert) → lädt die **Flash-Version**
- ⇒ JTAG-Image geht bei beiden verloren. Das FPGA neu konfigurieren lädt nie die JTAG-Firmware.
- Nach JTAG-Neukonfiguration gibt es **nicht immer** einen USB-Disconnect; der Host kann weiter
  mit dem alten Session-Zustand arbeiten (Device-Nummer bleibt). Für aussagekräftige Tests
  muss man eine **echte frische Enumeration** erzwingen.
- Host-seitiges `authorized 0/1` allein ist **nicht ausreichend** → führt zu
  `can't set config #1, error -32` (Stall), auch bei intakter Firmware.

### Bus-Reset ohne Stromverlust (erhält JTAG-Image!)
```python
# USBDEVFS_RESET auf das Geräte-Node senden (erzwingt echte Bus-Enumeration)
import fcntl, os, glob
for d in glob.glob('/sys/bus/usb/devices/*'):
    try:
        if open(d+'/idVendor').read().strip()=='1209' and open(d+'/idProduct').read().strip()=='ada1':
            bus = d.split('/')[-1].split('-')[0]
            node = f"/dev/bus/usb/{bus.zfill(3)}/{open(d+'/devnum').read().strip().zfill(3)}"
            break
fd = os.open(node, os.O_RDWR)
fcntl.ioctl(fd, 0x5514)   # USBDEVFS_RESET
```
Danach enumeriert das Gerät mit der JTAG-geladenen Firmware komplett neu.

## Ergebnisse

### adatface_EP4CGX150.sof — ✅ BESTANDEN
- Enumeration ok, ALSA-Karte `ADATfaceUSB1` mit **Playback + Capture** (je 2 Altsets: 2ch / 36ch, S32_LE, 48 kHz)
- 36ch-Capture-Stream öffnet und Gerät bleibt stabil (kein Disconnect)
- Überlebt `USBDEVFS_RESET` und frische Enumeration

### adatface_EP4CGX150_32chan_longIR.sof (2024-02-04) — ❌ FEHLGESCHLAGEN
- USB-Gerät enumeriert, ALSA-Karte registriert sich, **aber ohne jegliche PCM-Geräte**
  (nur `controlC0` + MIDI, kein `pcmC0D0p/c`, kein `stream0`)
- Ursache im dmesg (4× wiederholt):
  ```
  parse_audio_format_rates_v2v3(): unable to retrieve number of sample rates (clock 1)
  ```
  ⇒ die UAC2-Abfrage der Sample-Rates am Clock-Entity (bRequest NUMBER_OF_SAMPLE_RATES)
  wird vom Gateware beantwortet, aber die Antwort ist defekt/bleibt aus — der Treiber
  verwirft darauf alle Formate → keine PCM-Geräte.
- Genau das gemeldete Symptom „meldet sich nicht ordnungsgemäß als ALSA-Gerät".

### build_quartus23.1/top.sof (2024-02-08) — ❌ FEHLGESCHLAGEN (gleiches Bild)
### build/top.sof (2024-02-09, neueste Version) — ❌ FEHLGESCHLAGEN (gleiches Bild)
- Identisches Symptom: Karte registriert, keine PCM-Geräte, 4×
  `parse_audio_format_rates_v2v3(): unable to retrieve number of sample rates (clock 1)`
- Bei `build_quartus23.1/top.sof` zusätzlich beim ersten Verbindungsversuch:
  `device descriptor read/64, error -110` (Timeout) — danach Enumeration ok.

### ⇒ Fazit: Regression zwischen 01.02. und 04.02.2024
Nur der Build vom **2024-02-01** (der im Flash liegende) funktioniert. Alle drei späteren
Builds (04.02., 08.02., 09.02.) haben exakt dasselbe Fehlerbild in der UAC2-Clock-Entity-
Abfrage. Es liegt also eine zwischen dem 01.02. und dem 04.02. eingeführte Regression vor,
die in allen Folgebauten bestehen blieb. Prüfanfang: Änderungen im UAC2-Request-Handler
(Clock-Entity: NUMBER_OF_SAMPLE_RATES / SAMPLE_RATE_RANGE) und im USB-Deskriptor-Setup
aus diesem Zeitfenster (git log 01.02.–04.02.2024).

### Flash-Inhalt — identifiziert und gesichert
- Vollständiger Dump (16 MiB, read-only via openFPGALoader):
  `flash_backup_EP4CGX150/flash_backup_EP4CGX150_2026-11-13.bin`
  (md5 `59334eacce2d8b58ad912d6e084f6842`)
- Flash-Chip: Winbond W25Q128 (JEDEC 0xef4018), Daten 0x20–0x4B3243, Rest leer
- Identifikation: Bit-Reverse pro Byte des Dumps matcht **vollständig** (16 MiB)
  mit `adatface_EP4CGX150.jic` ab Datei-Offset 0x9E. `adatface_EP4CGX150_32chan_longIR.jic`
  matcht nicht (12 Byte Zufallstreffer).
- ⇒ **Im Flash läuft adatface_EP4CGX150 (Version vom 01.02.2024)** — die getestete
  funktionierende Version. Damit ist die Flash-Version identisch mit dem bestandenen Testfall.

## Auffälligkeiten (beide Versionen)
- dmesg: `config 1 interface 3 altsetting 0 bulk endpoint 0x3/0x83 has invalid maxpacket 64`
  (High-Speed-Bulk sollte 512 sein) — harmlos, aber auffällig (Diagnose-/ILA-Interface).

## Flash-Readback (wiederverwendbar, auch für Cyclone V)

1. `sudo apt-get install openfpgaloader` (v0.13.1)
2. Bridge-Bitstream `spiOverJtag_<PART>.rbf.gz` nach `/usr/share/openFPGALoader/`:
   - Für EP4CGX150DF27I7 gebaut aus [openFPGALoader-Quellen](https://github.com/trabucayre/openFPGALoader)
     (`spiOverJtag/altera_spiOverJtag.v` + `altclkctrl_inst.v`), Family auf
     **"Cyclone IV GX"** gepatcht, kompiliert mit `quartus_sh --flow compile`
     (Projekt in `tools/openfpgaloader_bridge/spiOverJtag.qsf`),
     `quartus_cpf -c --option=bitstream_compression=off` → RBF, dann `gzip -k`
   - Für Cyclone V (5CEFA2F23C8): gleiche Prozedur mit Family **"Cyclone V"** und
     DEVICE 5CEFA2F23C8 (das `ifdef cyclonev` im .v passt dann unverändert).
     Alternativ prüfen, ob `spiOverJtag_5ce423.rbf.gz` passt (wohl nicht exakt — 5CEFA4).
3. Dump (read-only):
   ```
   openFPGALoader --cable usb-blaster --fpga-part EP4CGX150DF27I7 \
     --dump-flash --file-size 16777216 -o 0 dump.bin
   ```
   (dauert ~10 min für 16 MiB; überschreibt die laufende FPGA-Konfiguration mit der Bridge
   → danach einmal RST-Knopf drücken, um die Flash-Version zurückzuladen)
4. **Vergleich mit .jic:** Der Dump ist **pro Byte bitinvers** (LSB/MSB vertauscht) zur
   Datenpage im .jic. Erst `bytes.translate()` mit Bit-Reverse-Tabelle anwenden, dann
   vergleichen. Beim `.jic` liegen die Daten ab Datei-Offset 0x9E (nach dem Header),
   Flash-Offset 0 ↔ jic-Offset 0x9E.
