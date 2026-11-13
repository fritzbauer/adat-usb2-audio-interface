#!/bin/bash
# Rebuild the project venv for the adat-usb2-audio-interface gateware.
#
# Background: the original venv was created on the old host (Gentoo, /home/user)
# with Python 3.11 and a hand-picked set of dependencies ("dependency tricks" —
# see VENV.md). Venvs are not relocatable, so it broke when the project moved to
# /mnt/Audio and only Python 3.13 is installed in the VM.
#
# This script rebuilds `venv` using the OFFLINE/TRICK path that has been proven
# to work end-to-end:
#   1. fresh venv with the newest system Python (3.13 works fine — all the
#      amaranth-family packages are pure Python)
#   2. copies the dependency versions that were pinned in the old venv from
#      venv-3.11-alt/lib/python3.11/site-packages (the old venv is kept as
#      venv-3.11-alt and also contains the editable source checkouts in src/)
#   3. rewrites the old host paths to the current mount point
#   4. installs the few ABI-bound packages (numpy etc.) fresh from PyPI
#
# The clean-from-scratch path (host, with internet) is documented in VENV.md.
#
# Usage:  ./initialize-python-environment.sh   (run from the gateware directory)
set -e
cd "$(dirname "$0")"

OLD="$PWD/venv-3.11-alt/lib/python3.11/site-packages"
NEW="$PWD/venv/lib/python3.13/site-packages"
OLDHOME=/home/user/Computer/Coden/Audio/adat-usb2-audio-interface
NEWHOME=/mnt/Audio/adat-usb2-audio-interface

if [ ! -d "$OLD" ]; then
    echo "ERROR: $OLD not found — the old venv backup is missing."
    echo "On a host with internet, use the pip-based setup documented in VENV.md instead."
    exit 1
fi

# 1. fresh venv (replace a broken/old one if present)
rm -rf venv
python3 -m venv venv
. venv/bin/activate

# 2. copy the pinned dependency versions: everything except venv tooling and
#    the ABI-bound packages (numpy/scipy/markupsafe use compiled C extensions
#    built for CPython 3.11; they are reinstalled via pip below)
(cd "$OLD" && cp -a $(ls | grep -v "^pip\|^setuptools\|^pipenv\|^tox\|^virtualenv\|^distlib\|^filelock\|^pluggy\|^pyproject_api\|^platformdirs\|^pkg_resources\|^cachetools\|^certifi\|^chardet\|^idna\|^requests\|^urllib3\|^click\|^colorama\|^docs\|^examples\|^numpy\|^scipy\|^markupsafe\|^MarkupSafe\|^numpy.libs\|^scipy.libs\|^yaml\|^charset_normalizer\|\.dist-info$\|\.egg-info$") "$NEW/")
(cd "$OLD" && for d in *.dist-info; do case "$d" in pip*|setuptools*|pipenv*|tox*|virtualenv*|PyYAML*|charset_normalizer*|numpy*|scipy*|MarkupSafe*|markupsafe*|apollo*|libusb1*|luna*) continue;; esac; [ -d "$NEW/$d" ] || cp -a "$d" "$NEW/"; done)

# 3. editable-install finders point at the old host paths; fix them
#    (the old venv directory was renamed to venv-3.11-alt)
sed -i "s|$OLDHOME|$NEWHOME|g; s|gateware/venv/src/|gateware/venv-3.11-alt/src/|g" \
    "$NEW"/__editable___*finder.py "$NEW"/*.pth 2>/dev/null || true

# 4. install the ABI-bound packages fresh from PyPI (python 3.13 wheels)
pip install -q numpy markupsafe pyyaml charset-normalizer pyvcd==0.2.4

# 5. smoke test
python - <<'EOF'
import amaranth
from luna.usb2 import USBDevice
from usb_descriptors import USBDescriptors
import amlib.stream, amlib.io.i2s, adat
from amaranth_boards.qmtech_5cefa2 import QMTech5CEFA2Platform
import amaranth_yosys
print("venv OK — amaranth", amaranth.__version__)
EOF
echo "Done. Use: . venv/bin/activate"
