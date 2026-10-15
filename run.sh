#!/bin/sh
set -eu
specimetry_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$specimetry_dir/.venv/bin/python" ]; then
    echo "Set up this tool's .venv as described in $specimetry_dir/README.md" >&2
    exit 2
fi
exec "$specimetry_dir/.venv/bin/python" -m specimetry "$@"
