# EtherADAT bring-up tools

Host-side helpers for debugging the EtherADAT bridge on real hardware,
saved into the repo so they survive VM reboots (which wipe `/tmp`).
Full context, status and next steps: `../ETHERADAT_HANDOFF_INPUT_20261123.md`
(one directory up, i.e. `gateware/`).

All are meant to run from `gateware/` with the project venv
(`venv/bin/python`), the ones using AF_PACKET need root:

| tool | needs root | purpose |
|------|-----------|---------|
| `parse_eduart.py` | (sudo cat) | decode the 250 ms UART status dump on `/dev/ttyUSB0` (AA 01 … 5A) |
| `uart_rates.py`   | (sudo cat) | compute per-second rates of the UART counters (rejects wrapping/aliasing) |
| `gen_tones.py`    | no  | create `/tmp/edtools/tone{16,out1,out2}.wav` (run after every reboot) |
| `slot_by_input.py`| yes | check TX payload slots == exact `EMPTY_USER_FRAME` (free-run) |
| `loop_sniff_tx.py`| yes | decode the PC's own playback frames (confirms the driver encodes the tone) |
| `loop_sniff_all.py`| yes | **main RX-debug tool**: decode IN1/IN2/IN3 from the FPGA TX frames while playing a tone |
| `loop_full.py`    | no  | play tone16 while capturing AdatNet + Audient (output path / full-duplex test) |

Common usage:

```bash
cd gateware
venv/bin/python etheradat/bringup_tools/gen_tones.py

# UART status
sudo stty -F /dev/ttyUSB0 115200 cs8 -cstopb -parenb raw -echo
timeout 3 sudo cat /dev/ttyUSB0 | venv/bin/python etheradat/bringup_tools/parse_eduart.py

# per-second rates of the same counters
timeout 3 sudo cat /dev/ttyUSB0 | venv/bin/python etheradat/bringup_tools/uart_rates.py

# ADAT-input decode (play OUT2=2 kHz, decode each input from Ethernet)
sudo venv/bin/python etheradat/bringup_tools/loop_sniff_all.py /tmp/edtools/tone_out2.wav

# TX payload sanity
sudo venv/bin/python etheradat/bringup_tools/slot_by_input.py
```

Defaults (override via env / args): interface `enxf8e43b907775`
(`ADATNET_IF`), AdatNet PCM `hw:1,0` (`ADATNET_PCM`), Audient `hw:0,0`
(`AUDIENT_PCM`), UART `/dev/ttyUSB0`, tones under `/tmp/edtools`.

Note: `aplay`/`arecord` are ALSA test tools (alsa-utils), not in the repo.
