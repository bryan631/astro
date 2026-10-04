#!/usr/bin/env bash
# Install the cedar-solve plate solver (imports as `tetra3`) into .venv.
# Pinned commit; --no-deps because upstream pins Pillow<9, which no longer builds and is not
# actually required (numpy/scipy/pillow come from our own pyproject dependencies).
set -euo pipefail
cd "$(dirname "$0")/.."
PIP=${PIP:-.venv/bin/pip}
$PIP install -q --no-deps "git+https://github.com/smroid/cedar-solve@607e3f8da7db0743bb6a26037e21c40b6befe49a"
