"""Regenerate report supplements from frozen inputs and saved final models."""
import os
from pathlib import Path
import subprocess
import sys

def main():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, MPLCONFIGDIR=str(Path('/private/tmp/stablecoin-matplotlib')))
    for module, args in [('modelling.build_corrected_dataset', []), ('modelling.reporting.event_table', []), ('modelling.reporting.detection_by_type', []), ('modelling.reporting.coverage_table', []), ('modelling.reporting.final_models', [])]:
        subprocess.run([sys.executable, '-W', 'ignore::DeprecationWarning', '-m', module, *args], cwd=root, env=env, check=True)
if __name__ == '__main__':
    main()
