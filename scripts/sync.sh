#!/bin/bash
# Sync .py notebook files to .ipynb for local development
# Jupytext only syncs files whose timestamps have changed (incremental)

cd "$(dirname "$0")/../src" || exit 1

for pyfile in *.py; do
    # Only sync files with jupytext cell markers (notebooks, not modules)
    if grep -q "^# %%" "$pyfile" 2>/dev/null; then
        jupytext --sync "$pyfile"
    fi
done
