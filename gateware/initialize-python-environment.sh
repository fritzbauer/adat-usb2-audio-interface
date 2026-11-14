#!/bin/bash
# Rebuild the project venv for the adat-usb2-audio-interface gateware.
#
# Details and background: see VENV.md. Key facts:
#  - amaranth is pinned to commit 597b1b8 (0.4.1-era code; pip reports the
#    version string as 0.5.0.dev0 since the metadata computation changed —
#    only the commit matters), amaranth-soc to 87ee8a5.
#  - amlib, amaranth-boards, usb-descriptors, usb-protocol are EDITABLE
#    installs from gateware/src/ — these checkouts carry local patches
#    (amlib test/i2c import fixes, qmtech_ep4cgx150 pin fixes) that are NOT
#    upstream. Don't refresh them from git without diffing first!
#  - luna (usb2-highspeed-core) is installed editable from
#    /mnt/Audio/usb2-highspeed-core; the M9K patch (read_port
#    transparent=False) is committed there (a9d5217). Alternatively the
#    pre-built patched egg from venv-3.11-alt can be used (see below).
#
# Usage:  ./initialize-python-environment.sh   (run from the gateware directory)
set -e
cd "$(dirname "$0")"

if [ ! -d src/amlib ]; then
    echo "ERROR: src/amlib not found — the pinned editable sources are missing."
    echo "They were reconstructed from git at pinned commits on 14.11.2026;"
    echo "see VENV.md for the exact commits."
    exit 1
fi

rm -rf venv
python3 -m venv venv
. venv/bin/activate

# 1. PyPI packages (plain, ABI-bound ones get matching wheels automatically)
pip install -q numpy markupsafe pyyaml charset-normalizer pyvcd==0.2.4 \
    pyserial pyusb construct packaging decorator networkx jinja2 \
    platformdirs filelock pluggy pyproject_api
pip install -q amaranth-yosys==0.35.0.0.post81 wasmtime==15.0.0

# 2. pinned amaranth-family packages (--no-deps: never let pip re-resolve)
pip install -q --no-deps \
    "amaranth @ git+https://github.com/amaranth-lang/amaranth.git@597b1b883924d8949061b52270eb55b97a7cfb76"
pip install -q --no-deps \
    "amaranth-soc @ git+https://github.com/amaranth-lang/amaranth-soc.git@87ee8a52d07a2f85b05a04db84644dce48fdfa23"

# 3. editable installs from the local patched checkouts
for p in src/amlib src/usb-descriptors src/python-usb-protocol src/amaranth-boards; do
    pip install -q --no-deps -e "$p"
done

# 4. luna: pre-built patched egg (proven path). The M9K patch is also
#    committed in /mnt/Audio/usb2-highspeed-core (a9d5217); installing luna
#    editable from that repo (pip install --no-deps -e ...) is a possible
#    alternative but is NOT the tested configuration.
if [ -f venv-3.11-alt/lib/python3.11/site-packages/luna-0.1.0.dev0-py3.11.egg ]; then
    cp venv-3.11-alt/lib/python3.11/site-packages/luna-0.1.0.dev0-py3.11.egg \
       venv/lib/python3.13/site-packages/
    echo "venv/lib/python3.13/site-packages/luna-0.1.0.dev0-py3.11.egg" \
        > venv/lib/python3.13/site-packages/luna_egg.pth
else
    echo "WARNING: luna egg not found in venv-3.11-alt; installing luna editable"
    echo "from the local repo (untested configuration, M9K patch is committed there)."
    pip install -q --no-deps -e /mnt/Audio/usb2-highspeed-core
fi

# 5. smoke test
python - <<'EOF'
import amaranth
import amlib.stream, amlib.io.i2s, adat
from luna.usb2 import USBDevice
from usb_descriptors import USBDescriptors
from amaranth_boards.qmtech_5cefa2 import QMTech5CEFA2Platform
import amaranth_yosys
print("venv OK — amaranth", amaranth.__version__)
EOF
echo "Done. Use: . venv/bin/activate"
