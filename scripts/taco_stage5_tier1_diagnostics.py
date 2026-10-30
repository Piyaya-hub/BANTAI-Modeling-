"""D074 CPU saved-query hierarchy audit, no model or photo execution."""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.taco_stage5_material_hierarchy import SCHEMES, MODES, prepare_rows, evaluate_case


def evaluate_views(data,rows,legacy_ids):
    by_image=defaultdict(list)
    for r in rows:by_image[r['image_id']].append(r)
    all_ids={i['id'] for i in data['images']}
    for scheme in SCHEMES:
        for mode in MODES:
            # Apply NMS once per image, then reuse unchanged retained rows in each view.
            kept=[r for ident in sorted(all_ids) for r in prepare_rows(by_image[ident],scheme,mode)]
            for view,ids in [('combined_65',all_ids),('legacy_47',legacy_ids),('annex_18',all_ids-legacy_ids)]:
                sub=dict(data,images=[i for i in data['images'] if i['id'] in ids],
                         annotations=[a for a in data['annotations'] if a['image_id'] in ids])
                result=evaluate_case(sub,[r for r in kept if r['image_id'] in ids],scheme,'raw')
                yield dict(result,view=view,taxonomy=scheme,nms=mode)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args()
    out=args.run_dir/'material-hierarchy';out.mkdir(exist_ok=False)
    paths=[args.run_dir/n for n in ('validation-epoch-1.json','soup_tier1_hr_blend2575.json','soup_tier1_hr_5050.json')]
    data=json.loads((ROOT/'exports/taco/strict-clean-stage5-v1/validation.json').read_text())
    legacy={i['id'] for i in json.loads((ROOT/'exports/taco/strict-clean-stage2-v1/validation.json').read_text())['images']}
    result=[];hashes={}
    for path in paths:
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        saved=json.loads(path.read_text());rows=saved['raw_queries']
        assert len(rows)==65*900 and len({(r['image_id'],r['query_id']) for r in rows})==len(rows)
        cases=[dict(c,state=path.stem) for c in evaluate_views(data,rows,legacy)]
        official=next(c for c in cases if c['view']=='combined_65' and c['taxonomy']=='9-specific' and c['nms']=='raw')
        for metric in ('AP','AP50','AP75','AR100'):
            assert abs(official['metrics'][metric]-saved['combined_65'][metric])<1e-12
        result.extend(cases)
        (out/(path.stem+'.json')).write_text(json.dumps(cases,indent=2))
        print(path.stem,'27 cases complete',flush=True)
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name] for p in paths)
    (out/'summary.json').write_text(json.dumps({'authorization':'D-074','diagnostic_only':True,
       'legacy_conditioning':'nine-class prompt subset, not separate eight-prompt evaluation',
       'input_sha256':hashes,'cases':result},indent=2))
    print(out,flush=True)

if __name__=='__main__':main()
