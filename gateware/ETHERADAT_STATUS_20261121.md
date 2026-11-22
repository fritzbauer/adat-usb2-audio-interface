# EtherADAT Baseline Bring-up — Stand 21.11.2026 (Session-Ende)

## Ziel
Stock-LiteEth UDP/IP-Baseline (migen) auf der ADATface (EP4CGX150) zum
Laufen bringen: ping 192.168.42.2 vom PC (ASIX-USB-NIC
enxf8e43b907775, PC=192.168.42.1) + UDP-Echo auf Port 1234.

## Was funktioniert (verifiziert)
- **Build-Flow**: Dateien in VM generieren
  (`venv/bin/python etheradat/scripts/generate_quartus_files_baseline.py
  --ref-clk-mode input <dir>`), nach `gateware/build` kopieren,
  `~/quartus_build_host.sh` baut auf dem Host, `quartus_pgm -c
  "USB-Blaster [1-1]" -m JTAG -o "P;top.sof"` flasht volatil.
  ACHTUNG: `LUNA_PLATFORM=platforms:ADATFaceCycloneIV_EP4CGX150` immer
  mitgeben (Default wurde auf EP4CGX150 geändert, aber setdefault
  reicht nicht, wenn env schon gesetzt war) — sonst baust du
  versehentlich Cyclone V und der Flash schlägt fehl (JTAG-ID-Mismatch).
- **Sim**: beide Tests GRÜN (test_baseline.py: ARP-Reply + UDP-Echo
  byte-exakt, mit dem realignierenden RX-Shim).
- **RX-Pfad auf Hardware Komplett beweisbar sauber** (UART-Probe):
  Kernel-Frames (IPv6-RS, DHCP, ARP) kommen byte-exakt inkl. SFD/FCS
  durch den RX-Shim in den plain-edge (NICHT invertierten) Domain.
- REF_CLK (vom Modul, REF_CLK-OUT-Strap), RXD0/1, CRS_DV: ok.
- MDC/MDIO ok (BMCR-Write aller 32 Adressen = 0x2100 landet, Link-Drop
  danach war der Beweis; MDIO-Force-FD-Instrument:
  etheradat/rmii_mdio_fd.py — aber für Baseline nicht nötig).

## Festgestellte Fehler — alle im Repo gefixt (uncommitted)
1. **LAN8720 assertiert CRS_DV 1-2 Zyklen NACH den ersten
   Preamble-Dibits** (nicht gleichzeitig wie im Sim) → SFD landet
   falsch ausgerichtet → Preamble-Checker wirft jeden Frame.
   Fix: **SFD-Realignment im RX-Shim**
   (etheradat/liteeth/rmii_shim.py RMIIRXShim: scannt auf SFD-Muster
   01_01_01_11, emittiert synthetische Preamble 7x55+d5, fängt Daten
   dahinter exakt ausgerichtet; 2-Entry-Byte-Queue). Sim-verifiziert,
   Hardware-verifiziert (Marker-01-Dumps sauber).
2. **RX-Sampling-Kante**: Modul ändert RXD an der FALLING edge →
   plain-edge (rx_negedge=False) ist korrekt; invertierte Kante gibt
   kaputte Frames (UART-Probe: pos-Shim sauber, neg-Shim Müll).
   Baseline-Wrapper: rx_negedge=False (Default).
3. **TX-Shim kombinatorische Pads**: Sink-Stream sauber, Pads korrupt
   (verstreute Dibit-Fehler, konstante Muster ok) → Output-Register
   eingebaut (txd_r/tx_en_r), ABER die Pad-Korruption blieb (Marker 03
   bleibt korrupt auch mit Registern und beiden Kanten) — UNGEKLÄRT,
   echter Hardware-TX-Verifikation ausstehend.
4. **UDP-Echo-Deadlock** (Sim-Fund): dw=8 sys-domain → keine per-Port
   FIFOs → unbuffered Echo blockt MAC-Crossbar → Fix SyncFIFO in der
   Echo-Loop + src/dst-Port-Swap.

## Der AKTUELLE Blocker (wo wir stehen)
LED-Isolation der Baseline (0,7-s-Blips, etheradat_baseline.py):
- usb2 (RX-Shim-Bytes) blippt ✓
- MAC-Core-Ausgang (sync4 = depacketizer.sink.valid) blippt ✓
- **Depacketizer sieht NIE valid&last (sync2 dunkel), rohes last
  (sync3) dunkel, CRC-Checker-Ausgang last dunkel, CDC-Ausgang
  last dunkel UND first dunkel (letzter Build!)**
- MAC-Depacketizer output stumm → Crossbar Dispatch never matches
  0x0806 → ARP nie → Stack sendet nie (sync1/USB-TX dunkel).

Letzter Build (build/, EP4CGX150, 21:05): LED-Messung CDC-out
last/first — Ergebnis stand noch aus (USB2-Sync4-Blink-Bestätigung
wartet). CDC-Ausgang ohne last UND first = die
LiteEthMACCDC (Stufe 1 der rx_datapath.pipeline) frisst die
Frame-Rahmen-Flags. Verdacht: die CDC-FIFO-Flags (last/first) gehen
bei der axi-stream CDC im LiteX-Master verloren, ODER die Pipeline
liegt in einer anderen Domain als angenommen.

## Nächste Schritte
1. **LED-Isolation fortsetzen**: build/ enthält CDC-out last/first
   LEDs. Wenn beide dunkel: LiteEthMACCDC-Verkettung prüfen
   (rx_datapath.pipeline[1]), vgl. liteeth/mac/core.py RXDatapath —
   ggf. dump CDC-Interna über UART (depacketizer-uart-Dump ist schon
   in gen_baseline.py: per-Frame [0xAA][header hi][header lo], Trigger
   dep.sink.valid&last — feuert nie, weil last fehlt!).
   Nächster sinnvoller Dump: CDC asyncfifo we/re/level-Signale.
2. **PDB-Hinweis**: sim läuft ewig (debug_etdump.py, 450k Zyklen,
   abgebrochen) — zu groß für migen-Sim; stattdessen debug_
   first_frame.py-artige kurze Traces oder die LED-/UART-Route.
3. **TX-Verifikation**: Das TX-Pad-Korruptions-Rätsel (Marker 03)
   bleibt offen; Multimeter-Durchgangstest der TX-Lanen (J2:8/10/12 +
   Nachbarn) war laut User ok. Nächster Schritt: TXGEN-Frames im
   tcpdump sehen (Modul-LED-Flackern deutete auf TX-Aktivität).
4. **UART-Infrastruktur**: gen_uartprobe.py / rmii_uartprobe.py /
   generate_quartus_files_uartprobe.py (--tx-swap --tx-no-negedge
   Flags), ttyUSB0 @115200, Dump-Format:
   [0xAA][marker][len][bytes], heartbeat 0x7E; Marker 01=pos-RX,
   02=neg-RX, 03=TX-Pads, 04=TX-Sink.
5. Sim-Umgebung: gepinnte Commits (migen e19524c, litex 6438a25,
   liteeth a54c730) + migen-Sim-Patch (patches/
   migen-e19524c-sim-write-only-port.patch). Scratch-venv wurde nach
   /mnt/Audio/adat-usb2-audio-interface/etheradat-scratch-saved/
   gesichert (liteenv + git-clones + debug-skripte); Debug-Skripte
   erwarten /tmp/etheradat-scratch/liteenv — nach VM-Start symlink
   oder Pfad anpassen.

## Infrastructure-Notizen
- USB-Blaster: jtagd als root nötig (`sudo /opt/altera_lite/25.1std/
  quartus/bin/jtagd`) — user-mode jtagd blockt mit port-permissions.
- ASIX-NIC: ethtool autoneg verriegelt ("Operation not supported"),
  Link = 100 Mbit HALF — für Baseline unkritisch (ARP-Antwort kommt
  ms-nach-dem-Request), nur der Instant-Echo-Repeater jammt.
- UART: `sudo stty -F /dev/ttyUSB0 115200 raw -echo` vor jedem Lesen.
- build.old* Verzeichnisse = Build-Historie, unimportant, git-untracked.
- gen_baseline.py hat jetzt: ARP-Ketten-LEDs, MAC-Stufen-LEDs,
  UART-Depacketizer-Dump (alles Debug, temporär).
