#!/usr/bin/env python3
"""Run the whole EtherADAT test suite (project venv, from gateware/).

    ./venv/bin/python -m etheradat.tests.run_tests [-v]

The LiteEth-side RMII loopback test lives separately (needs the
migen/litex scratch venv): etheradat/liteeth/test_rmii_shim.py
"""

import sys
import unittest

from etheradat.tests import (test_reference, test_adat_frame_codec,
                             test_discovery)

MODULES = [test_reference, test_adat_frame_codec, test_discovery]


def main():
    verbosity = 2 if "-v" in sys.argv else 1
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(loader.loadTestsFromModule(m)
                               for m in MODULES)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
