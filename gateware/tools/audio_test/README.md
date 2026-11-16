# ADATface Audio-Path HW-Test — 14.11.2026

Erster Hardware-Test der **Audioinhalte** (bisher wurde nur die USB/ALSA-Ebene
getestet, s. `firmware_test_report_2026-11-13*.md`).

## Setup (Benutzer-Loopback-Verkabelung)

- ADATface **Out 1-8** → Audient iD14 ADAT-Eingang, erscheint dort als Capture-Kanäle **3-10**
- ADATface **Out 9-16** → ADATface **In 9-16** (optischer Loopback, TOSLINK-Port 2)

## Testwerkzeug

`tools/audio_test/audio_loopback_test.py` (mit `gateware/venv/bin/python` ausführen):

- **Test A** (Ausgabepfad): `aplay` 36ch @ ADATface, gleichzeitig `arecord` 10ch @ iD14
- **Test B** (Eingabepfad): `aplay` 36ch @ ADATface, gleichzeitig `arecord` 36ch @ ADATface

Testsignal pro Kanal `c`: selbstidentifizierende Rampe
`v(n, c) = ((n & 0x3FFF) << 9) | (c << 4)` (24-bit-sicher, S32_LE).

Wichtige Erkenntnisse zur Signalpfad-Semantik:
- Das Gerät behält nur die **oberen 24 bit** des 32-bit-Worts (left-justified).
  Der Kanal-Tag unter Bit 8 wird also maskiert; die Rampe (`n << 9`) überlebt.
- `n = 0` liest sich als 0 zurück (Truncation) — kein Defekt.
- iD14 und ADATface haben unabhängige Clocks → gelegentliche dup/drop-Paare
  im iD14-Capture sind normal (Clock-Resync, Nettodrift 0).

## Ergebnisse

### Baseline: funktionierende Flash-Firmware (adatface_fir6, 2022-06-01)

Beide Tests **PASS**. Daten bit-exakt (nach 24-bit-Maske). Nur:

- Start-Glitch bei t ≈ 0.042 s: +1 dupliziertes + 1 veraltetes Sample (TX-Pfad,
  von beiden Empfängern identisch gesehen; stale-Wert variiert pro Run)
- seltene dup/drop-Paare ~1-2/s (Capture-Seite, Nettodrift 0)

### Aktueller Stand (HEAD) vor Fix

Beide Tests **FAIL, stark korrupt**: Test A ~200-450 Ereignisse/s,
Test B Loopback-Kanäle 9-16 nur Nullen (RX synchronisiert nie). iD14 empfängt
strukturierten Müll statt der Rampe.

### Ursache (gefunden per Diff der generierten top.v)

**amlib-Commit `9a43a4a` (27.01.2022) hat den NRZIEncoder vom `adat`- in den
`sync`-Domain verschoben** (`m.d.adat` → `m.d.sync` in
`amlib/utils/nrziencoder.py`). Das gepinnte adat-core (700d2ad) instanziiert
den Encoder ohne DomainRenamer, daher lief der NRZI-Encoder im sync-Domain
(61.44 MHz) statt im adat-Bitclock-Domain (12.288 MHz) → **ADAT-Ausgangs-
Bitstream mit 5-facher Bitrate** → kein Empfänger (iD14, eigener RX) konnte
synchronisieren. Die USB/ALSA-Ebene blieb sauber (deshalb fielen die
Enumerationstests vom 13.11. nicht darauf auf).

Referenz: `build_orig/top.v` (2022-02-12) enthält `always @(posedge adat_clk)`
im NRZI-Encoder-Modul; der 14.11.-Build davor enthielt `always @(posedge clk)`.

### Fix

`gateware/src/amlib/amlib/utils/nrziencoder.py` zurück auf `m.d.adat`
(= Verhalten der funktionierenden 2022-Builds), dokumentiert in
`patches/amlib-db885c9-local.patch` (regeneriert).

### Nach dem Fix (Build 14.11.2026, volatil geladen)

Beide Tests **PASS** (2 aufeinanderfolgende Läufe), Fehlerbild identisch zur
Baseline (Start-Glitch + ~1-1.5 dup/drop-Paare/s, alle ≤ 2 Samples).

## Logs

- `tools/audio_test/log_2026-11-14_fixed_NRZI_cycloneV.log` — letzte PASS-Läufe

Der Flash wurde zu keinem Zeitpunkt beschrieben; alle Builds nur volatil per
JTAG geladen (`quartus_pgm -m jtag -o "p;..."`).
