#!/bin/bash
# Build a fully self-contained project venv for the adat-usb2-audio-interface
# gateware. Everything needed lives in this repository (gateware/src,
# gateware/vendor) plus PyPI/git network access — no external venv backup
# required.
#
# Details and background: see VENV.md. Key facts:
#  - amaranth is pinned to commit 597b1b8 (0.4.1-era code; pip reports the
#    version string as 0.5.0.dev0 since the metadata computation changed —
#    only the commit matters), amaranth-soc to 87ee8a5.
#  - amlib, amaranth-boards, usb-descriptors, usb-protocol are EDITABLE
#    installs from gateware/src/ — these checkouts carry local patches
#    (amlib test/i2c import fixes, qmtech_ep4cgx150 pin fixes) that are NOT
#    upstream. Don't refresh them from git without diffing first!
#  - luna is installed from the vendored patched egg in gateware/vendor/
#    (M9K patch: read_port transparent=False; also committed in
#    /mnt/Audio/usb2-highspeed-core as a9d5217).
#  - Everything installed with --no-deps where pinning matters: pip must
#    never re-resolve amaranth (it would pull 0.4.5/0.6 from PyPI and break
#    the patched amlib).
#
# Usage:  ./initialize-python-environment.sh [VENV_DIR]   (default: venv)
set -e
cd "$(dirname "$0")"

VENV_DIR="${1:-venv}"
if [ "$VENV_DIR" != "venv" ]; then
    echo "NOTE: installing into non-default directory '$VENV_DIR'"
fi

if [ ! -d src/amlib ]; then
    echo "ERROR: src/amlib not found — the pinned editable sources are missing."
    echo "They are gitignored; reconstruct from git at the pinned commits"
    echo "documented in VENV.md."
    exit 1
fi
if [ ! -f vendor/luna-0.1.0.dev0-py3.11.egg ]; then
    echo "ERROR: vendor/luna-0.1.0.dev0-py3.11.egg not found — vendored patched"
    echo "luna egg is missing."
    exit 1
fi

rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"

SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"

# 1. PyPI packages (ABI-bound ones get matching wheels automatically)
pip install -q numpy markupsafe pyyaml charset-normalizer pyvcd==0.2.4 \
    pyserial pyusb construct packaging decorator networkx jinja2 \
    platformdirs filelock pluggy pyproject_api
pip install -q amaranth-yosys==0.35.0.0.post81 wasmtime==15.0.0

# 2. pinned amaranth-family packages (--no-deps: never let pip re-resolve)
pip install -q --no-deps \
    "amaranth @ git+https://github.com/amaranth-lang/amaranth.git@597b1b883924d8949061b52270eb55b97a7cfb76"
pip install -q --no-deps \
    "amaranth-soc @ git+https://github.com/amaranth-lang/amaranth-soc.git@87ee8a52d07a2f85b05a04db84644dce48fdfa23"
# adat-core: proven working state is 700d2ad (repo HEAD 49092bd is newer,
# not the tested configuration)
pip install -q --no-deps \
    "adat @ git+https://github.com/amaranth-community-unofficial/adat-core.git@700d2ad"

# 3. editable installs from the local patched checkouts
for p in src/amlib src/usb-descriptors src/python-usb-protocol src/amaranth-boards; do
    pip install -q --no-deps -e "$p"
done

# 4. luna from the vendored patched egg
cp vendor/luna-0.1.0.dev0-py3.11.egg "$SITE/"
echo "$SITE/luna-0.1.0.dev0-py3.11.egg" > "$SITE/luna_egg.pth"

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
echo "Done. Use: . $VENV_DIR/bin/activate"
