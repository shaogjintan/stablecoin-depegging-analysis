#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "Running Midterm Presentation Pipeline..."
echo "========================================================================"
python3 07_coin_filter_and_sensitivity.py
echo "========================================================================"
echo "Done! Figures, tables, and defense notes are updated."
echo "========================================================================"
