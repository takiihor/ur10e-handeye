#!/usr/bin/env bash
# Set up this checkout (isolated .venv, pinned packages, sibling repos) and run ur_handeye.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_DIR"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

for sibling in charuco-generator charuco-realsense; do
    if [[ ! -d "$PROJECT_DIR/../$sibling" ]]; then
        printf 'Error: ../%s is missing; clone it next to this repository.\n' "$sibling" >&2
        exit 1
    fi
done

if [[ ! -x "$VENV_PYTHON" ]]; then
    SELECTED_PYTHON="${PYTHON_BIN:-}"
    if [[ -z "$SELECTED_PYTHON" ]]; then
        for candidate in python3.12 python3.11 python3.13 python3; do
            if command -v "$candidate" >/dev/null 2>&1 && \
               "$candidate" -c 'import sys; raise SystemExit(not (3, 11) <= sys.version_info[:2] <= (3, 13))'; then
                SELECTED_PYTHON="$candidate"
                break
            fi
        done
    fi
    if [[ -z "$SELECTED_PYTHON" ]]; then
        printf 'Error: install Python 3.11-3.13 (pyrealsense2 and ur_rtde wheels), or set PYTHON_BIN.\n' >&2
        exit 1
    fi
    printf 'Creating .venv with %s...\n' "$SELECTED_PYTHON"
    "$SELECTED_PYTHON" -m venv "$PROJECT_DIR/.venv"
fi

STAMP="$PROJECT_DIR/.venv/.requirements.sha256"
WANT="$(cat requirements.txt pyproject.toml | sha256sum | cut -d' ' -f1)"
if [[ ! -f "$STAMP" || "$(cat "$STAMP")" != "$WANT" ]]; then
    printf 'Installing pinned dependencies...\n'
    "$VENV_PYTHON" -m pip install --disable-pip-version-check --quiet -r requirements.txt
    "$VENV_PYTHON" -m pip install --disable-pip-version-check --quiet --no-deps \
        -e ../charuco-generator -e ../charuco-realsense -e .
    printf '%s' "$WANT" > "$STAMP"
fi

if [[ $# -eq 0 ]]; then
    set -- ui
fi
exec "$VENV_PYTHON" -m ur_handeye "$@"
