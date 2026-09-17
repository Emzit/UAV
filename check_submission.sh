#!/bin/sh
# Validate a Swarm Rescue submission (workspace tree or teamNNN_evalstep.zip).

set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1

if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "Error: no Python interpreter found (.venv/bin/python or python3)." >&2
  exit 1
fi

exec "$PYTHON" scripts/check_submission.py "$@"
