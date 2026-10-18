"""Independently reconstruct and validate a TACO preparation against current sources."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.taco_preparation import validate_preparation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='Preparation directory containing preparation_manifest.json')
    args = parser.parse_args(argv)
    try:
        result = validate_preparation(args.dataset)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result['passed'] else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'TACO validation failed ({type(exc).__name__}): {exc}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
