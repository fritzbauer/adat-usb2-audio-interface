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

# Python version guard: Python 3.14 is supported since the tracer patch
# (patches/amaranth-597b1b8-local.patch restores automatic port naming;
# without it every port comes out as an anonymous "\$signal" identifier
# and Quartus synthesis fails). Hard-fail only on untested versions.
PYV="$(python3 -c 'import sys; print(sys.version_info[0]*100 + sys.version_info[1])')"
if [ "$PYV" -ge 315 ]; then
    echo "ERROR: python3 is $(python3 -V). Gateware compatibility is only"
    echo "verified up to Python 3.14 (port-naming inference). Use 3.13 or 3.14."
    exit 1
fi
PYTHON3=python3

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
$PYTHON3 -m venv "$VENV_DIR"
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

# 2a. apply the amaranth patches (e.g. the Python 3.14 port-naming fix in
# tracer.py). Same idempotency scheme as the adat patches below.
for pf in patches/amaranth-597b1b8-local.patch; do
    if patch -d "$SITE" --dry-run -p1 -s < "$pf" > /dev/null 2>&1; then
        patch -d "$SITE" -p1 -s < "$pf" && echo "applied $(basename "$pf")"
    elif patch -d "$SITE" --dry-run -R -p1 -s < "$pf" > /dev/null 2>&1; then
        echo "$(basename "$pf") already applied"
    else
        echo "ERROR: $(basename "$pf") neither applies nor reverse-applies to $SITE/amaranth"
        exit 1
    fi
done

# 2b. apply all local patches from patches/ to the freshly installed
# adat-core in site-packages. Idempotent: if a patch no longer applies
# forward but applies in reverse, it is already in place and skipped.
ADAT_PKG="$SITE/adat"
for pf in patches/adat-*700d2ad-local.patch; do
    if patch -d "$SITE" --dry-run -p1 -s < "$pf" > /dev/null 2>&1; then
        patch -d "$SITE" -p1 -s < "$pf" && echo "applied $(basename "$pf")"
    elif patch -d "$SITE" --dry-run -R -p1 -s < "$pf" > /dev/null 2>&1; then
        echo "$(basename "$pf") already applied"
    else
        echo "ERROR: $(basename "$pf") neither applies nor reverse-applies to $ADAT_PKG"
        echo "(the pinned adat-core source may have changed upstream)"
        exit 1
    fi
done

# 3. editable installs from the local patched checkouts
# verify they still carry their local patches (documented in patches/)
if ! grep -q "FFSynchronizer" src/amlib/amlib/io/i2c.py; then
    echo "ERROR: src/amlib is missing its local patch (i2c.py should use FFSynchronizer)"
    echo "see patches/amlib-db885c9-local.patch; the checkout may have been refreshed"
    exit 1
fi
if ! grep -q 'PinsN("A25", dir="o")' src/amaranth-boards/amaranth_boards/qmtech_ep4cgx150.py; then
    echo "ERROR: src/amaranth-boards is missing its local patch"
    echo "(qmtech_ep4cgx150.py pin dir= fixes; see patches/amaranth-boards-627badc-local.patch)"
    echo "the checkout may have been refreshed from git"
    exit 1
fi
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

# 6. verify the adat patches landed (the fixed decoder's characteristic
# constant must be present) and that port naming is intact
if ! grep -q "bit_time - 1" "$SITE/adat/nrzidecoder.py"; then
    echo "ERROR: adat patch did not land (no bit_time-1 wrap in nrzidecoder.py)"
    exit 1
fi
if ! grep -q "LOAD_FAST_BORROW" "$SITE/amaranth/tracer.py"; then
    echo "ERROR: amaranth patch did not land (no LOAD_FAST_BORROW in tracer.py)"
    exit 1
fi
# elaborate an ADAT receiver/transmitter and check the emitted ports are
# NAMED (anonymous \$signal ports mean the amaranth naming inference
# failed, which Quartus synthesis then rejects)
python - <<'EOF'
import sys
from amaranth import Module
from adat import ADATReceiver, ADATTransmitter
from amaranth.back import verilog

m = Module()
r0 = ADATReceiver(98304000)
r1 = ADATTransmitter()
m.submodules.r0 = r0
m.submodules.r1 = r1
m.d.comb += [
    r0.adat_in.eq(0),
    r1.sample_in.eq(0),
    r1.valid_in.eq(0),
    r1.last_in.eq(0),
    r1.user_data_in.eq(0),
]
src = verilog.convert(m, name="top",
                      ports=[r0.adat_in, r0.sample_out, r1.adat_out])
bad = src.count("\\$signal")
if bad:
    print(f"ERROR: port naming failed: {bad} anonymous \\$signal ports in emitted Verilog")
    print("this happens on Python >= 3.14; use Python 3.13 or older")
    sys.exit(1)
print("port naming OK (no anonymous $signal ports)")
EOF
echo "Done. Use: . $VENV_DIR/bin/activate"
