"""Explicit local functional experiment; not unittest discovery or model training.

Run from the repository root using the verified .venv. Writes only ignored JSON
and logs, keeps the owner's configuration byte-for-byte, and never downloads.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.configuration import load_config, strict_load
from scripts.results import validate_result
from scripts.gui_viewer import section_data
from scripts.viewer import passing_candidates


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    config_path = root / 'config/detection.json'
    baseline = load_config(config_path)
    originals = sorted(Path(baseline['paths']['input']).glob('*'))
    originals = [p for p in originals if p.is_file()]
    result_root = Path(baseline['paths']['results'])
    preserved = {str(p): digest(p) for p in [config_path, *originals, *result_root.rglob('*.json')]}
    study = result_root / ('phase2e-' + uuid.uuid4().hex)
    study.mkdir()
    plan = [
        ('A-baseline', {}),
        ('B-concrete', {'prompt': 'crumpled plastic bag. empty drink pouch. cigarette butt. candy wrapper. discarded paper.'}),
        ('C-categories', {'prompt': 'a garbage pile. an individual discarded item. a group of individual litter.'}),
        ('D-box-only', {'threshold': 0.35}),
        ('E-text-only', {'text_threshold': 0.25}),
    ]
    report = {'study': study.name, 'baseline': baseline, 'preserved_sha256': preserved, 'experiments': []}
    (study / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    for name, changes in plan:
        config = copy.deepcopy(baseline)
        config.update(changes)
        config['paths']['results'] = str(study / name)
        path = study / (name + '.json')
        path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        command = [sys.executable, 'scripts/detect_images.py', '--mode', 'infer',
                   '--config', str(path), '--local-files-only', '--no-display']
        with (study / (name + '.log')).open('w', encoding='utf-8') as log:
            code = subprocess.call(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        item = {'name': name, 'changes_from_baseline': changes, 'command': command,
                'exit_code': code, 'runs': []}
        for run_path in sorted((study / name).glob('*/run.json')):
            run = strict_load(run_path)
            rows = []
            for filename in run['records']:
                record_path = run_path.parent / filename
                result = validate_result(strict_load(record_path))
                source = result['source']
                assert digest(Path(source['image_id'])) == source['sha256']
                data = section_data(result)
                passing = passing_candidates(result)
                assert [r[0] for r in data['summary']] == [d['id'] for d in passing]
                assert set(r[0] for r in data['tokens']) <= set(d['id'] for d in passing)
                rows.append({'path': str(record_path), 'source': source, 'status': result['status'],
                             'display': result['display'], 'error': result['error'],
                             'timing': result['timing'], 'device': result['inference']['actual_device'],
                             'detections': passing, 'review': result['review'],
                             'raw_queries': len((result.get('query_diagnostics') or {}).get('queries', [])),
                             'passing_tokens': len(data['tokens'])})
            assert all(p.suffix == '.json' for p in run_path.parent.iterdir())
            item['runs'].append({'manifest': str(run_path), 'run': run, 'images': rows,
                                 'environment': strict_load(run_path.parent / 'environment.json')})
        report['experiments'].append(item)
        assert all(digest(Path(p)) == h for p, h in preserved.items()), 'Preexisting input/result/config changed'
        (study / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(name, 'exit', code, 'runs', [r['run']['run_id'] for r in item['runs']], flush=True)
        if code:
            print('Stopped after observed failure; inspect log. No settings were selected.', flush=True)
            break
    print('Study:', study, flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
