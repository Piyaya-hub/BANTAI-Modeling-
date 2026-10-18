"""Run bounded synthetic diagnostics without patching or replacing native loss."""
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from scripts.loss_investigation import main

if __name__=='__main__':
    raise SystemExit(main())
