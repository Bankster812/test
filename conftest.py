"""Root conftest — ensures the repo root is importable so tests can
`import wholesale` / `import neuromorphic` regardless of pytest's rootdir
insertion behaviour."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
