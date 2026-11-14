# Wie kommt man an ein funktionierendes Python-venv für das Gateware?

Stand: 13.11.2026 (Cyclone-V-Rebuild). Kontext: `firmware_test_report_2026-11-13.md`,
`firmware_test_report_2026-11-13_cycloneV.md`.

## Warum das kompliziert ist (die „Dependency-Tricks")

Das Design nutzt Amaranth **0.4.1.dev1+g597b1b8** (Commit-Nov. 2023) mit einer
handverlesenen Kombination aus (alle aus `venv-3.11-alt/lib/python3.11/site-packages`
ablesbar, exakt dort gepinnt):

| Paket | Version | Anmerkung |
|---|---|---|
| amaranth | 0.4.1.dev1+g597b1b8 | aus git, Commit `597b1b883924d8949061b52270eb55b97a7cfb76` (vor dem Entfernen von `compat`) |
| amaranth-soc | 0.1.dev80+g87ee8a5 | gepinnt (s. INSTALL-Rezept im `FIR`-Branch) || amaranth-boards | git-Snapshot (editable, `venv-3.11-alt/src/amaranth-boards`) | **lokal gepatcht**: `qmtech_ep4cgx150.py` (und `qmtech_5cefa2.py`) |
| amlib | 0.1.dev163+gdb885c9 (editable, `venv-3.11-alt/src/amlib`) | **lokal gepatcht**: `amlib/io/i2c.py`, `amlib/test/__init__.py` |
| adat | 0.1.dev109+g700d2ad | aus `/mnt/Audio/adat-core` |
| luna (usb2-highspeed-core) | 0.1.0.dev0 (egg, Nov. 2023) | Repo: `/mnt/Audio/usb2-highspeed-core` |
| usb_descriptors / usb_protocol | 0.1.1 / 0.0 (editable, `venv-3.11-alt/src/`) |usb-descriptors-Lieferstand |
| amaranth-yosys | 0.35.0.0.post81 | yosys als `yosys.wasm`, läuft über `wasmtime` (kein System-yosys nötig) |
| wasmtime / ziglang | 15.0.0 / 0.8.1 | WASM-Runtime für yosys |
| numpy | 1.26.3 | nur `np.zeros` + Indexierung → **numpy 2.x funktioniert auch** |

Auf einem neuen System `pip install` mit den **heutigen git-HEADs** (wie
`requirements.txt` es tut) bricht, weil sich amlib/amaranth-boards/amaranth-soc
seither weiterbewegt haben und luna geändert wurde — deshalb die lokale Kopie.

## Weg 1 (VM/offline, bewährt): venv aus dem Alt-Bestand rekonstruieren

Der alte venv (`venv-3.11-alt`) wurde auf dem alten Host unter `/home/user/Computer/Coden/Audio/…`
mit Python 3.11 angelegt und ist nicht-relocatable. Er wird nur noch als
**Abbild der Pinning-Entscheidungen** benutzt (`site-packages` + `src/`-Checkouts).

```bash
cd gateware
./initialize-python-environment.sh    # baut venv neu auf, wie oben beschrieben
. venv/bin/activate
```

Das Skript:
1. legt ein frisches venv mit dem System-Python an (Python **3.13 funktioniert** —
   der Amaranth-Stack ist reines Python; die kleinen ABI-Unterschiede
   (numpy/markupsafe/PyYAML) werden frisch installiert bzw. haben pure-Python-Fallbacks),
2. kopiert die gepinnten Pakete aus `venv-3.11-alt/lib/python3.11/site-packages`
   (ohne pip/setuptools und die ABI-gebundenen Pakete),
3. schreibt die alten Host-Pfade (`/home/user/Computer/Coden/Audio/…`) auf
   `/mnt/Audio/…` um und fixiert die `venv-3.11-alt/src`-Pfade der editable
   Installs (die Sourcen der lokalen Patches liegen in `venv-3.11-alt/src/`,
   **dort nicht aufräumen!**),
4. installiert `numpy markupsafe pyyaml charset-normalizer pyvcd==0.2.4` frisch,
5. macht einen Import-Smoke-Test.

## Weg 2 (Host, mit Internet): sauber per pip

Entspricht dem INSTALL-Rezept aus dem `FIR`-Branch (Commit `16e177a`), nur
amaranth/amaranth-soc gepinnt statt `requirements.txt` (das fährt auf
bewegliche git-HEADs):

```bash
python3.11 -m venv venv          # 3.11 ist die erprobte Version; 3.13 ging in der VM auch
. venv/bin/activate
pip install --upgrade pip
# amaranth exakt wie im bewährten Stand (0.4.1.dev1, vor dem compat-Entfernen):
pip install git+https://github.com/amaranth-lang/amaranth.git@597b1b883924d8949061b52270eb55b97a7cfb76
pip install git+https://github.com/amaranth-lang/amaranth-soc.git@87ee8a52d07a2f85b05a04db84644dce48fdfa23
# amlib und amaranth-boards MIT den lokalen Patches installieren (nicht aus git!):
pip install -e venv-3.11-alt/src/amlib
pip install -e venv-3.11-alt/src/amaranth-boards
pip install git+https://github.com/amaranth-community-unofficial/adat-core.git
pip install venv/lib/python3.11/site-packages/luna-0.1.0.dev0-py3.11.egg   # oder -e /mnt/Audio/usb2-highspeed-core
pip install numpy pyvcd==0.2.4
```

Wichtig: **nicht** `pip install -r requirements.txt` — die git-URLs zeigen auf
heutige HEADs; der letzte Stand, der damit baute, ist Ende 2023/Anfang 2024.
Die lokalen Patches in `venv-3.11-alt/src/{amlib,amaranth-boards}` sind Teil der
funktionierenden Kombination.

## Quartus-Dateien generieren (ohne Synthese — z. B. in der VM)

```bash
cd gateware
. venv/bin/activate
./venv/bin/python scripts/generate_quartus_files.py build_cycloneV
```

Das läuft die Elaboration wie `adat_usb2_audio_interface.py` durch
(`LUNA_PLATFORM` steuert das Board; aktuell `platforms:ADATFaceCycloneV`)
und schreibt `top.v`, `top.qsf`, `top.sdc`, `top.qpf`, `build_top.sh` etc.
nach `build_cycloneV/` — **ohne** Quartus aufzurufen
(`platform.build(..., do_build=False)` + `plan.execute_local(..., run_script=False)`).

### Auf dem Host synthesen

```bash
cd gateware/build_cycloneV
PATH=/opt/altera_lite/25.1std/quartus/bin:$PATH ./build_top.sh
#  → top.sof (und per quartus_cpf/jic-Workflow die .jic zum Flashen)
```

Falls 25.1std keine Cyclone-V-Device-DB mehr hat, mit Quartus 23.1 bauen
(gleiche `build_top.sh`; Quartus-Pfad entsprechend anpassen — s. auch
`scripts/quartusPath.sh` für die alte Host-Notation).

### Testen (wie in den Berichten)

```bash
quartus_pgm -m jtag -o "p;top.sof"      # nur volatil! Flash nie beschreiben
# danach USB-Reset erzwingen, lsusb/aplay/arecord prüfen
# siehe tools/firmware_test/test_one.sh
```

## luna-Egg-Patch (2026-11-13, Teil der Tricks)

`venv/lib/python3.13/site-packages/luna-0.1.0.dev0-py3.11.egg` enthält zwei
lokale Patches für M9K-Inferenz (Quartus LAB-Problem, s. Bericht):

- `luna/gateware/memory.py`: `memory.read_port()` → `read_port(transparent=False)`
- `luna/gateware/usb/usb2/transfer.py`: `read_port(domain="usb")` →
  `read_port(domain="usb", transparent=False)`

Das gepatchte Egg liegt **auch** in `venv-3.11-alt/lib/python3.11/site-packages/`,
somit reproduziert `initialize-python-environment.sh` den Patch automatisch.
Quelle der Patches ist inzwischen auch das luna-Repo selbst
(`/mnt/Audio/usb2-highspeed-core`, Commit `a9d5217`) — bei einem Neu-Bau von luna
aus dem Repo sind sie damit enthalten; das Egg muss dann nicht mehr gepatcht
werden, solange das Egg aus der gepatchten Quelle stammt.
Backup: `/tmp/eggpatch/egg_new` (+ `egg_orig` = Original). Beim Neu-Zusammensetzen
des Eggs per `zipfile`: nur die **erste** Instanz jedes Namens behalten.

## luna-Pin (WICSLIG beim Aktualisieren)

`/mnt/Audio/usb2-highspeed-core` ist bewusst am Nov-2023-Snapshot (`13acd44`,
+ M9K-Patch-Commit `a9d5217`) gepinnt. `origin/main` war am 13.11.2026 129
Commits voraus (amaranth-Pins etc. geändert) — **nicht pullen** ohne Neutest
des Gateware-Builds; sonst bricht die „Tricks"-Kombination aus VENV.md.

## Update 14.11.2026 — venv-3.11-alt gelöscht & restauriert, neue Source-Heimat `gateware/src/`

`venv-3.11-alt/` wurde am 13.11. versehentlich gelöscht (dadurch fehlten alle
editable Sourcen → `No module named 'amlib'`) und am 14.11. aus einem **älteren
Backup** restauriert. Konsequenzen:

- Die lokal gepatchten Editable-Sourcen leben jetzt **kanonisch in
  `gateware/src/`** (gitignore): `amlib` (db885c9 + i2c.py/test-Patches),
  `amaranth-boards` (627badc + qmtech_ep4cgx150-Patch), `usb-descriptors`
  (5cc4faf), `python-usb-protocol` (538983d). Patches wurden aus dem restaurierten
  Backup rüberkopiert — die Backup-Sourcen sind identisch mit dem Stand von
  gestern (nur das luna-Egg darin war alt und wurde erneut gepatcht).
- `initialize-python-environment.sh` baut das venv jetzt **netz-basiert** auf:
  PyPI-Pakete + gepinnte amaranth/amaramanth-soc-Commits (`--no-deps`, nie
  resolvieren lassen!) + `pip install --no-deps -e gateware/src/*` + luna-Egg
  aus `venv-3.11-alt` (gepatcht). Der alte "site-packages kopieren"-Weg ist obsolet.
- Merkwürdigkeit: `pip` meldet für amaranth-Commit `597b1b8` heute
  `0.5.0.dev0` statt `0.4.1.dev1` (Metadaten-Berechnung upstream geändert) —
  entscheidend ist der **Commit-Hash**, `direct_url.json` bestätigt ihn.

## Update 14.11.2026 (Abend) — Setup ist jetzt self-contained

`initialize-python-environment.sh` baut ein vollständiges venv ohne
`venv-3.11-alt` auf:

- gepatchtes luna-Egg ist **vendored**: `gateware/vendor/luna-0.1.0.dev0-py3.11.egg`
- `adat` wird aus git am erprobten Commit `700d2ad` installiert
  (Repo-HEAD `49092bd` ist neuer = ungetestet)
- optionaler Ziel-Parameter: `./initialize-python-environment.sh /tmp/testvenv`
  (für Wegwerf-Tests)

Vollständiger Selftest bestanden: frisches venv von Null + Generatorlauf.
**`venv-3.11-alt/` kann gelöscht werden.** Zu erhalten bleiben: `gateware/src/`
(gitignored, lokal gepatchte Sourcen) und `gateware/vendor/` (im Repo).
