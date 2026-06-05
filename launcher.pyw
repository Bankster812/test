"""
Keystone Property Partners — launcher
======================================
Double-click this file (or set it as a Desktop shortcut target) to open the
company dashboard without any terminal window.

Works on:
  Windows  — run with pythonw.exe (no console)
  Mac/Linux— run with python3
"""

import os
import subprocess
import sys
import time
import webbrowser
import urllib.request
import urllib.error

PORT = int(os.environ.get("WS_DASHBOARD_PORT", "8200"))
ROOT = os.path.dirname(os.path.abspath(__file__))

# Start the server as a background process
server = subprocess.Popen(
    [sys.executable, "-m", "wholesale.run", "--port", str(PORT)],
    cwd=ROOT,
)

# Wait up to 20 s for the HTTP server to be ready
for _ in range(40):
    try:
        urllib.request.urlopen(f"http://localhost:{PORT}/api/state", timeout=1)
        break
    except (urllib.error.URLError, OSError):
        time.sleep(0.5)

# Open the dashboard in the default browser
webbrowser.open(f"http://localhost:{PORT}")

# Keep this process alive so the server stays running
try:
    server.wait()
except KeyboardInterrupt:
    server.terminate()
    server.wait()
