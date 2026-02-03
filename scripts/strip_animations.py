#!/usr/bin/env python3
"""
Strip matplotlib animations from executed Jupyter notebooks.

This script finds cells containing HTML(anim.to_jshtml()) outputs and replaces
them with a placeholder message. This keeps notebook file sizes manageable
for publishing while preserving all other outputs.

Usage:
    python scripts/strip_animations.py notebook.ipynb [output.ipynb]
    python scripts/strip_animations.py --all  # Process all .ipynb in current dir
"""

import argparse
import json
import re
import sys
from pathlib import Path


# Markers that identify matplotlib jshtml animations
ANIMATION_MARKERS = [
    "/* Instantiate the Animation class. */",
    "requestAnimationFrame",
    'class="animation"',
    "anim.frame_seq",
]

# Minimum size (bytes) for an output to be considered a potential animation
MIN_ANIMATION_SIZE = 50000

PLACEHOLDER_HTML = """\
<div style="padding: 20px; background: #f0f0f0; border: 1px solid #ccc; border-radius: 5px; text-align: center;">
    <p style="margin: 0; color: #666;">
        <strong>Animation removed for publishing.</strong><br>
        Run this notebook locally to see the interactive animation.
    </p>
</div>
"""


def is_animation_output(output: dict) -> bool:
    """Check if a cell output is a matplotlib jshtml animation."""
    if output.get("output_type") not in ("execute_result", "display_data"):
        return False

    data = output.get("data", {})
    html = data.get("text/html", "")

    # Join if it's a list of strings
    if isinstance(html, list):
        html = "".join(html)

    # Check size first (animations are large)
    if len(html) < MIN_ANIMATION_SIZE:
        return False

    # Check for animation markers
    for marker in ANIMATION_MARKERS:
        if marker in html:
            return True

    return False


def strip_animations_from_notebook(notebook: dict) -> tuple[dict, int]:
    """
    Strip animation outputs from a notebook.

    Returns:
        Tuple of (modified notebook, number of animations stripped)
    """
    count = 0

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        new_outputs = []
        for output in cell.get("outputs", []):
            if is_animation_output(output):
                # Replace with placeholder
                new_outputs.append({
                    "output_type": "display_data",
                    "data": {
                        "text/html": PLACEHOLDER_HTML,
                        "text/plain": ["Animation removed for publishing."]
                    },
                    "metadata": {}
                })
                count += 1
            else:
                new_outputs.append(output)

        cell["outputs"] = new_outputs

    return notebook, count


def process_notebook(input_path: Path, output_path: Path | None = None) -> int:
    """
    Process a single notebook file.

    Returns:
        Number of animations stripped
    """
    if output_path is None:
        output_path = input_path

    with open(input_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    notebook, count = strip_animations_from_notebook(notebook)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Strip matplotlib animations from Jupyter notebooks"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input notebook file (or --all to process all .ipynb files)"
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output notebook file (default: overwrite input)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all .ipynb files in current directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for processed notebooks (with --all)"
    )

    args = parser.parse_args()

    if args.all:
        notebooks = list(Path(".").glob("*.ipynb"))
        if not notebooks:
            print("No .ipynb files found in current directory")
            return 1

        output_dir = args.output_dir or Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)

        total_count = 0
        for nb_path in notebooks:
            output_path = output_dir / nb_path.name
            count = process_notebook(nb_path, output_path)
            total_count += count
            if count > 0:
                print(f"{nb_path.name}: stripped {count} animation(s)")
            else:
                print(f"{nb_path.name}: no animations found")

        print(f"\nTotal: {total_count} animation(s) stripped from {len(notebooks)} notebook(s)")
        return 0

    if not args.input:
        parser.print_help()
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1

    output_path = Path(args.output) if args.output else None
    count = process_notebook(input_path, output_path)

    dest = output_path or input_path
    print(f"Stripped {count} animation(s) from {input_path}")
    if output_path:
        print(f"Output written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
