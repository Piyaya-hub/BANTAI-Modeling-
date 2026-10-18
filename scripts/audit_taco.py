"""Audit local TACO without modifying source images, annotations or detector."""
import argparse
from pathlib import Path
import sys
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.taco_dataset import DEFAULT_DATASET, DEFAULT_TAXONOMY, audit_dataset, write_reports


def add_source_arguments(parser):
    parser.add_argument('--dataset-root', default=str(DEFAULT_DATASET))
    parser.add_argument('--annotations', help='COCO JSON; default <dataset-root>/annotations.json')
    parser.add_argument('--taxonomy', default=str(DEFAULT_TAXONOMY))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument('--output', default='reports/taco/'+uuid.uuid4().hex, help='New report directory; never overwrites')
    args = parser.parse_args(argv)
    try:
        audit = audit_dataset(args.dataset_root, args.annotations, args.taxonomy)
        folder = write_reports(audit, args.output)
        counts = audit['summary']['counts']
        print('Images:', counts['image_status'])
        print('Usable annotations (all / eight mapped classes):', counts['usable_annotations_all_categories'], '/', counts['usable_mapped_annotations'])
        for row in audit['summary']['classes']:
            print(f"{row['label']}: {row['usable_objects']} objects in {row['usable_unique_images']} images")
        print('Excluded annotations require an explicit handling decision. Report:', folder)
        return 1 if counts['invalid_annotations'] or counts['image_issue_records'] or audit['summary']['structural_issues'] else 0
    except (OSError, ValueError) as exc:
        print(f'TACO audit failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
