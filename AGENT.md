# AGENT.md — Hinweise für Coding-Agenten (Claude etc.)

## Builds: nur über den Host-Synthese-Weg

Quartus-Synthese/-Fitting darf **nicht** in dieser VM ausgeführt werden —
nur über den Host:

```bash
~/quartus_build_host.sh        # ssh auf den Quartus-Host (User/IP stehen im Skript)
```

Die Verbindung zum Host startet direkt den Build des Projekts in
`gateware/build/` (dort liegt die zu buildende `build_top.sh` + `top.v/qsf/sdc`).
Der Quartus-Flow (map/fit/asm/sta) läuft auf dem Host; die Ergebnisse erscheinen
im geteilten Mount unter `gateware/build/` (`top.sof`, `top.fit.summary`,
`top.sta.summary`, …). Auf den Host-SSH-Sitzungen **kein** Interaktives starten
und nicht herumprobieren, wenn etwas nicht auf Anhieb klappt — dem Nutzer Bescheid
geben.

Dateien für den Build generiert der Agent selbst (das ist OK und billig):
```bash
cd gateware && . venv/bin/activate
./venv/bin/python scripts/generate_quartus_files.py <build_dir>
```
…oder mit dem Board-Auswahl-Env (`LUNA_PLATFORM`, s. `gateware/adat_usb2_audio_interface.py`
und `gateware/platforms.py`). Dann `build_cycloneV…` nach `gateware/build/`
kopieren (alte Inhalte ggf. vorher sichern) und den Host-Build triggern.

## Keine speicherintensiven Jobs in der VM

Diese VM hat wenig RAM. **Verboten/OOM-gefahr:**
- yosys-Synthese über das gesamte Design (`synth_intel_alm`, `synth`,
  `read_verilog` + lang laufende Passes) — einmal mit OOM abgestürzt
- jede andere Full-Design-Synthese/Simulation (auch `amaranth_yosys` über
  komplette netlists)

**Erlaubt/billig:**
- Amaranth-Elaboration + Quartus-Dateierzeugung (`scripts/generate_quartus_files.py`)
- Datei-Inspektion der generierten Verilog-Outputs (grep/wc) als "Verifikation"
- kurze, gezielte Tests der Python-Umgebung

Quartus-**Programmierung** (JTAG, volatil) ist in der VM in Ordnung und Teil des
etablierten Testflusses:

```bash
cd <build_dir> && /opt/altera_lite/25.1std/quartus/bin/quartus_pgm -m jtag -o "p;top.sof"
# danach: tools/firmware_test/usb_reset.py (sudo), dann lsusb / arecord / aplay
```

**Niemals den Flash beschreiben** — nur volatil ins FPGA-SRAM laden
(s. `gateware/firmware_test_report_2026-11-13*.md`).

## Sonstiges

- Python-Umgebung: `gateware/initialize-python-environment.sh`, Details/Deps in
  `gateware/VENV.md` (gepinnte luna-Snapshot-Quelle, nicht pullen).
- Ergebnisse von Tests (ALMs, Timing, USB/ALSA-Checks) in die Bericht-MDs oder
  an den Nutzer melden statt still zu bleiben.
