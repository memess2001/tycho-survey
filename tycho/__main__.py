"""``python -m tycho``: the command-line interface (:mod:`tycho.cli`).

The guard matters: survey workers are started with the ``spawn`` method, which re-imports the main module in every
child; there it must not run the CLI again.
"""
from tycho.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
