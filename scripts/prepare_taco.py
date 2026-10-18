"""Create validated, strict-clean COCO splits referencing external source images."""
import argparse
from pathlib import Path
import sys
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.configuration import resolve_path
from scripts.taco_preparation import DEFAULT_POLICY, fresh_audit, load_policy, make_plan, write_preparation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', default='reports/taco/local-review', help='Existing audit; source records are freshly validated')
    parser.add_argument('--policy', default=str(DEFAULT_POLICY))
    parser.add_argument('--output', default='exports/taco/'+uuid.uuid4().hex, help='New ignored output directory; images are not copied')
    parser.add_argument('--seed', type=int, help='Default from policy: 42')
    parser.add_argument('--ratios', type=float, nargs=3, metavar=('TRAIN', 'VALIDATION', 'TEST'), help='Default .70 .15 .15')
    args = parser.parse_args(argv)
    try:
        output = resolve_path(args.output)
        if output.exists(): raise FileExistsError(f'Output exists; choose a new directory: {output}')
        policy_path = resolve_path(args.policy); policy = load_policy(policy_path)
        print('Revalidating source COCO and image readability...', flush=True)
        audit = fresh_audit(args.audit)
        ratios = dict(zip(('train', 'validation', 'test'), args.ratios)) if args.ratios else None
        plan = make_plan(audit, policy, args.seed, ratios, progress=lambda message: print(message, flush=True))
        stats = write_preparation(audit, policy, policy_path, plan, output, args.audit)
        print('Strict-clean images / annotations:', stats['eligible_images'], '/', stats['eligible_annotations'])
        print('Splits:', stats['splits'])
        print('Near-duplicate candidates coassigned when eligible:', stats['near_duplicate_candidate_pairs'])
        print('Validated COCO export (licensing/trainer/coverage review still pending):', output)
        return 0
    except (OSError, ValueError) as exc:
        print(f'TACO preparation failed: {exc}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
