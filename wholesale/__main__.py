"""Allow `python -m wholesale` as a shortcut for `python -m wholesale.run`."""

from __future__ import annotations

import sys

from .run import main

if __name__ == "__main__":
    sys.exit(main())
