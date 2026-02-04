#!/bin/bash
# Sync .py notebook files to .ipynb and re-execute if changed
#
# Usage:
#   ./scripts/sync.sh              # Sync and execute only changed notebooks
#   ./scripts/sync.sh --no-execute # Sync only, skip execution
#   ./scripts/sync.sh --force      # Sync and execute ALL notebooks

set -e
SCRIPT_DIR="$(dirname "$0")/../"
cd "$SCRIPT_DIR/../src" || exit 1

# Kill all child processes on Ctrl+C
trap 'echo " Interrupted, killing background jobs..."; kill $(jobs -p) 2>/dev/null; exit 1' INT

MODE="auto"
if [[ "$1" == "--no-execute" ]]; then
    MODE="no-execute"
elif [[ "$1" == "--force" ]]; then
    MODE="force"
fi

# First pass: sync all notebooks (sequential, fast)
declare -a TO_EXECUTE=()

for pyfile in *.py; do
    # Only sync files with jupytext cell markers (notebooks, not modules)
    if grep -q "^# %%" "$pyfile" 2>/dev/null; then
        notebook="${pyfile%.py}.ipynb"

        # Capture jupytext output to check if file changed
        output=$(jupytext --sync "$pyfile" 2>&1)
        echo "$output"

        # Decide whether to execute
        if [[ "$MODE" == "force" ]]; then
            TO_EXECUTE+=("$notebook")
        elif [[ "$MODE" == "auto" ]] && echo "$output" | grep -q "Updating '$notebook'"; then
            TO_EXECUTE+=("$notebook")
        fi
    fi
done

# Second pass: execute notebooks in parallel
if [[ ${#TO_EXECUTE[@]} -gt 0 ]]; then
    echo ""
    echo "Executing ${#TO_EXECUTE[@]} notebook(s) in parallel..."

    for notebook in "${TO_EXECUTE[@]}"; do
        (
            echo "  Starting: $notebook"
            TQDM_DISABLE=1 jupyter nbconvert --to notebook --execute --inplace \
                --ExecutePreprocessor.timeout=1800 \
                --ExecutePreprocessor.kernel_name=python3 \
                "$notebook" && echo "  Done: $notebook" || echo "  Failed: $notebook"
        ) &
    done

    # Wait for all background jobs
    wait
    echo "All executions complete."
fi

# Strip animations and copy to root
echo ""
echo "Stripping animations..."
python "$SCRIPT_DIR/strip_animations.py" --all

echo ""
echo "Copying notebooks to root..."
cp *.ipynb "$SCRIPT_DIR/.."

echo "Done."
