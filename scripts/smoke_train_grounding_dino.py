"""Entry point for isolated, synthetic-only Grounding DINO qualification."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from scripts.training_smoke import main

if __name__ == '__main__':
    raise SystemExit(main())
