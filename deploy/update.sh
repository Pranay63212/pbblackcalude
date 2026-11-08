#!/usr/bin/env bash
# Pull the latest code, reinstall requirements, run the tests, and restart the
# scanner ONLY if every test passes (AGENTS.md rule 1). Otherwise the running
# version keeps going untouched.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO_DIR/.venv/bin/python"
cd "$REPO_DIR"
git pull --ff-only
"$PY" -m pip install -q -r requirements.txt
if "$PY" tests.py >/tmp/pbblake-tests.log 2>&1; then
  tail -3 /tmp/pbblake-tests.log
  sudo systemctl restart pbblake-scanner
  echo "Updated and restarted."
else
  tail -20 /tmp/pbblake-tests.log
  echo "TESTS FAILED - the scanner was NOT restarted; the previous version is still running."
  echo "The new files are on disk, so a crash or reboot would start them: send these errors to fix before then."
  exit 1
fi
