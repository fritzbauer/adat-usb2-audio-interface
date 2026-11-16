#!/usr/bin/env python3
"""Erzwingt eine frische USB-Enumeration des ADATface (1209:ada1) via USBDEVFS_RESET."""
import fcntl, os, glob, sys

node = None
for d in glob.glob('/sys/bus/usb/devices/*'):
    if ':' in d.split('/')[-1]:
        continue  # Interface-Verzeichnisse überspringen
    try:
        if (open(d + '/idVendor').read().strip() == '1209'
                and open(d + '/idProduct').read().strip() == 'ada1'):
            bus = d.split('/')[-1].split('-')[0]
            devnum = open(d + '/devnum').read().strip()
            node = f"/dev/bus/usb/{bus.zfill(3)}/{devnum.zfill(3)}"
            break
    except OSError:
        continue

if not node:
    print("NO_DEVICE")
    sys.exit(1)

fd = os.open(node, os.O_RDWR)
try:
    fcntl.ioctl(fd, 0x5514)  # USBDEVFS_RESET
    print(f"RESET_OK {node}")
except OSError as e:
    print(f"RESET_FAIL {node}: {e}")
    sys.exit(2)
finally:
    os.close(fd)
