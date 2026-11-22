# EtherADAT Baseline Bring-up — Stand 22.11.2026

## Ziel

Stock-LiteEth UDP/IP-Baseline auf der ADATface (EP4CGX150, LAN8720-Modul)
betreiben: ARP, `ping 192.168.42.2` und UDP-Echo auf Port 1234.

## Aufbau / Reproduktion (VM → Host)

1. Verilog generieren (Scratch-venv, **nicht** adatface-venv):
   ```
   /tmp/etheradat-scratch/liteenv/bin/python \
     etheradat/liteeth/gen_baseline.py
   ```
   Scratch ist bei VM-Neustart nötig:
   `ln -s /mnt/Audio/adat-usb2-audio-interface/etheradat-scratch-saved /tmp/etheradat-scratch`.
2. Quartus-Dateien erzeugen — `LUNA_PLATFORM` **immer explizit** setzen:
   ```
   LUNA_PLATFORM=platforms:ADATFaceCycloneIV_EP4CGX150 \
     venv/bin/python etheradat/scripts/generate_quartus_files_baseline.py \
     --ref-clk-mode input <build_dir>
   ```
3. Dateien nach `gateware/build/` kopieren, `build/top.sof` löschen,
   `~/quartus_build_host.sh` (kein direktes ssh!), dann volatil flashen:
   ```
   sudo /opt/altera_lite/25.1std/quartus/bin/jtagd
   /opt/altera_lite/25.1std/quartus/bin/quartus_pgm \
     -c 'USB-Blaster [1-1]' -m JTAG -o 'P;build/top.sof'
   ```
4. PC-Seite:
   ```
   sudo nmcli device set enxf8e43b907775 managed no
   sudo ip addr add 192.168.42.1/24 dev enxf8e43b907775
   sudo ip link set enxf8e43b907775 up
   ```
   Das Modul **muss Strom haben**, sonst gibt es keinen REF_CLK und damit
   weder Link noch UART-Ausgabe.

## Verifizierte Hardware-Fixes (22.11.2026)

Alle Änderungen sind in `etheradat/liteeth/{rmii_shim.py,gen_baseline.py,
fixed_depacketizer.py}`, `etheradat/etheradat_baseline.py` und
`etheradat/scripts/generate_quartus_files_baseline.py`.

1. **RMII-RX-Shim**: SFD-Scan wegen später CRS_DV-Assertion (21.11.), plus
   Re-Arm des Scanners nach jedem Frame. Ohne Re-Arm gab der Shim ab dem
   zweiten Frame das Preamble als Payload aus (UART-Byte-Taps).
2. **RX-CDC**: `LiteEthMACCDC` las auf Hardware `last` immer als 0 zurück.
   `sys` und `eth_rx` hängen am selben REF_CLK-Pin → `SameClockRXCDC`
   (Register-Slice). Dasselbe für `sys → eth_tx` bei `tx_negedge=False`.
3. **Depacketizer-Off-by-one**: LiteX' `Depacketizer` benutzt ein
   registriertes `sr_shift` im selben Edge wie das Header-Shiftregister;
   dadurch fehlte der erste Header-Byte. `fixed_depacketizer.py` macht das
   Signal kombinational. Für MAC und ARP nötig (via Monkeypatch).
4. **TX-Shim**: alte Pair/ emitting-FSM verlor ersten/letzten Dibit pro
   Byte. Neue 4-Dibit-Schieberegister-FSM mit lückenloser Byte-Übergabe.
5. **Wrapper**: `o_rmii_tx_en`/`o_rmii_txd` waren nie verbunden (top.v trieb
   die Pins konstant 0). Jetzt an `rmii.tx_en.o` / `Cat(txd0.o,txd1.o)`.
6. **SDC**: `rmii_0__ref_clk__io` ist jetzt mit 20 ns constrainiert. Ohne
   diese Constraint war das Depacketizer-Verhalten place&route-abhängig
   (ARP manchmal erkannt, manchmal nicht).
7. **UDP-Echo**: SyncFIFO im Loop (Sonst Crossbar-Deadlock bei dw=8) und
   Quell-/Zielport-Tausch.

Debug-Infrastruktur: `etheradat/liteeth/gen_txtest.py` + `etheradat/txtest.py`
senden einen festen Ethernet-Frame direkt durch den TX-Shim; damit wurde
der TX-Pfad unabhängig vom Stack verifiziert (Frame kam im tcpdump an).
Der UART-Dump des Baseline-Cores liefert alle ~170 ms
`[0xAA][8 Zähler]`: shim_last, cdc_last, dep_last, arp_req, arp_rep,
tx_last, rx_bytes, tx_bytes.

## Messergebnis (finaler Build)

- Link 100 Mbit HALF, `ip neigh` zeigt `192.168.42.2 lladdr 02:ad:6e:65:74:01 REACHABLE`.
- tcpdump zeigt ARP-Requests vom PC und **ARP-Replies vom FPGA**.
- UDP-Echo Port 1234 wird beantwortet.
- **Offen**: Echo-Payload ist um 1 Byte verschoben (z. B. `hello-etheradat`
  → `\xc9hello-etherada`), und ICMP-Echo-Replies haben denselben 1-Byte-
  Shift, weshalb `ping` trotz sichtbarer Replies keine Antwort zählt.
  Die Verschiebung liegt im IP-/ICMP-Payload-Pfad (Header/Payload-Grenze),
  nicht mehr in RMII, CDC, ARP oder TX-Pads. UART-Zähler am finalen Build:
  `shim_last=cdc_last=dep_last`, `arp_req/arp_rep` steigen, `tx_last` steigt.
- Sim (`test_baseline.py`: ARP-Reply + UDP-Echo) läuft mit den Fixes grün;
  der 1-Byte-Shift tritt nur auf Hardware auf.

## Nächste Schritte

1. Payload-Shift im LiteEth IP/UDP/ICMP-Pfad lokalisieren (UART-Byte-Taps
   an `ip_rx.source`, `udp_rx.depacketizer.source`, `icmp_echo.sink`),
   vermutlich eine weitere registrierte Valid/Data-Fehlausrichtung.
2. Danach Ping und byte-exakten UDP-Echo verifizieren und dieses Dokument
   aktualisieren.
