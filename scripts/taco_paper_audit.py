"""D069 metadata-only paper proposal. No photos, test.json, model or export.

Replays the original eligibility algorithm with cached image fingerprints and
asserts exact baseline ledger reproduction. JSON evidence goes to system temp.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from collections import Counter,defaultdict
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
PAPER_NAMES=('Normal paper','Magazine paper','Wrapping paper','Tissues','Paper bag','Paper cup',
             'Corrugated carton','Other carton','Drink carton','Meal carton','Egg carton',
             'Pizza box','Paper straw','Toilet tube')


def main():
    from scripts.configuration import strict_load
    from scripts.taco_preparation import make_plan,load_policy,RULES
    from scripts.taco_dataset import load_taxonomy
    from scripts.taco_splits import Groups,assign_splits
    from PIL import Image
    stage=ROOT/'exports/taco/strict-clean-stage2-v1';review=ROOT/'reports/taco/local-review'
    out=Path(tempfile.mkdtemp(prefix='bantai-d069-paper-'))
    print('Audit evidence directory: '+str(out),flush=True)
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.name=='test.json' or p.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff','.safetensors','.pt','.bin'):
                raise AssertionError('Photo/test/checkpoint access prohibited: '+str(p))
            flags=args[2] if len(args)>2 else 0
            if isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
                if not p.is_relative_to(out):raise AssertionError('Only temporary audit evidence may be written: '+str(p))
    sys.addaudithook(guard)
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
    manifest=strict_load(stage/'preparation_manifest.json');source=Path(manifest['source']['annotations'])
    inputs=[source,stage/'preparation_manifest.json',stage/'image_eligibility.jsonl',stage/'annotation_decisions.jsonl',
            stage/'duplicate_review.json',stage/'policy_snapshot.json',stage/'taxonomy_snapshot.json',stage/'statistics.json',
            review/'audit.json',review/'images.jsonl',review/'annotation_review.jsonl',ROOT/'config/taco_taxonomy.v1.json',
            ROOT/'config/taco_preparation_policy.v1.json',*[ROOT/'scripts'/n for n in
            ('train_taco.py','taco_training_data.py','taco_evaluation.py','training_checkpoint.py','training_adapter.py','training_smoke.py')]]
    before={str(p):sha(p) for p in inputs}
    for p in inputs:
        if p.parent==stage and p.name in manifest['artifacts']:assert sha(p)==manifest['artifacts'][p.name],p
    raw=strict_load(source);summary=strict_load(review/'audit.json');taxonomy=load_taxonomy()
    assert (len(raw['images']),len(raw['annotations']))==(1500,4784)
    reconstructed=hashlib.sha256(json.dumps(raw).encode()).hexdigest()
    assert before[str(source)]==manifest['source']['sha256'] or reconstructed==manifest['source']['sha256']
    assert before[str(ROOT/'config/taco_taxonomy.v1.json')]==manifest['source']['taxonomy_sha256']
    images=lines(review/'images.jsonl');annotations=lines(review/'annotation_review.jsonl');old=lines(stage/'image_eligibility.jsonl')
    assert [r['original'] for r in images]==raw['images']
    assert [r['original'] for r in annotations]==raw['annotations']
    categories={c['id']:c for c in raw['categories']}
    assert all(r['original_category']==categories[r['original_category']['id']] for r in annotations if r['original_category'])
    fingerprints={r['source_image_id']:r['fingerprint'] for r in old if r['fingerprint']}
    audit={'summary':summary,'taxonomy':taxonomy,'images':images,'annotations':annotations,'source_document':raw}
    policy=load_policy();assert policy==strict_load(stage/'policy_snapshot.json')
    def cached(row):
        value=fingerprints[row['original']['id']]
        assert value['file_sha256']==row['sha256'] and value['aspect_ratio']==row['actual_width']/row['actual_height']
        return copy.deepcopy(value)
    def blocked(*a,**kw):raise AssertionError('Photo decoding is prohibited')
    with patch('scripts.taco_preparation.fingerprint',side_effect=cached),patch.object(Image,'open',blocked):
        baseline=make_plan(audit,policy)
    assert baseline['eligibility']==old,'Cached baseline image ledger differs'
    assert baseline['annotation_decisions']==lines(stage/'annotation_decisions.jsonl'),'Baseline annotation ledger differs'
    assert sum(r['eligible'] for r in old)==315
    names=list(PAPER_NAMES);by_name={c['name']:c['id'] for c in raw['categories']};assert set(names)<=set(by_name)
    paper_ids={by_name[n] for n in names};rawpaper=[a for a in raw['annotations'] if a['category_id'] in paper_ids]
    proposed=copy.deepcopy(audit);proposed['taxonomy']['version']='2.0.0-proposal'
    proposed['taxonomy']['labels'].append({'id':8,'name':'paper','taco_categories':names})
    for row in proposed['annotations']:
        if row['original']['category_id'] in paper_ids:row.update(bantai_label='paper',bantai_label_id=8,handling='mapped')
    with patch('scripts.taco_preparation.fingerprint',side_effect=cached),patch.object(Image,'open',blocked):
        plan=make_plan(proposed,policy)
    oldids={r['source_image_id'] for r in old if r['eligible']}
    eligible={r['source_image_id'] for r in plan['eligibility'] if r['eligible']}
    paperimages={a['image_id'] for a in rawpaper};kept=paperimages&eligible;newids=eligible-oldids
    assert oldids<=eligible and newids==kept
    paper_by_image=Counter(a['image_id'] for a in rawpaper);by_image=defaultdict(list)
    for a in proposed['annotations']:by_image[a['original']['image_id']].append(a)
    category_rows=[]
    for name in names:
        rows=[a for a in rawpaper if a['category_id']==by_name[name]];keptrows=[a for a in rows if a['image_id'] in eligible]
        category_rows.append({'name':name,'category_id':by_name[name],'raw_boxes':len(rows),
            'raw_images':len({a['image_id'] for a in rows}),'strict_boxes':len(keptrows),'strict_images':len({a['image_id'] for a in keptrows})})
    waterfall=[];remaining=set(paperimages)
    for rule in RULES:
        removed={r['source_image_id'] for r in plan['eligibility'] if r['primary_reason']==rule}&remaining
        remaining-=removed
        waterfall.append({'rule':rule,'paper_images_removed':len(removed),'paper_boxes_removed':sum(paper_by_image[i] for i in removed),
            'paper_images_remaining':len(remaining),'paper_boxes_remaining':sum(paper_by_image[i] for i in remaining)})
    assert remaining==kept
    unlabeled={i for i in paperimages if any(a['handling']=='unlabeled_requires_decision' for a in by_image[i])}
    other={i for i in paperimages if any(a['bantai_label'] is None and a['handling']!='unlabeled_requires_decision' for a in by_image[i])}
    cooccurrence={'paper_images_with_unlabeled':len(unlabeled),'paper_images_with_other_excluded_excluding_unlabeled':len(other),
        'paper_images_with_both':len(unlabeled&other),'paper_images_with_any_excluded':len(unlabeled|other),
        'paper_boxes_on_any_excluded_images':sum(paper_by_image[i] for i in unlabeled|other),
        'overlapping_flags_not_additive':True}
    excluded_categories=[]
    for category in raw['categories']:
        values=[a['original'] for i in paperimages for a in by_image[i]
                if a['bantai_label'] is None and a['original']['category_id']==category['id']]
        if values:excluded_categories.append({'name':category['name'],'boxes':len(values),'images':len({a['image_id'] for a in values})})
    # Append-only split proposal: original315 assignments stay fixed; duplicate
    # groups touching an old split are anchored to it before new-group allocation.
    locked={r['source_image_id']:r['split'] for r in old if r['eligible']};groups=Groups(eligible)
    for group in plan['exact_duplicate_groups']:
        valid=[i for i in group if i in eligible]
        for i in valid[1:]:groups.join(valid[0],i)
    for pair in plan['near_duplicate_candidates']:
        a,b=pair['image_id_a'],pair['image_id_b']
        if a in eligible and b in eligible:groups.join(a,b)
    assignments=dict(locked);free=[];forced=[]
    for group in groups.members():
        old_splits={locked[i] for i in group if i in locked};assert len(old_splits)<=1
        new=[i for i in group if i in newids]
        if not new:continue
        if old_splits:
            split=next(iter(old_splits));assignments.update({i:split for i in new});forced.append({'ids':new,'split':split})
        else:free.append(new)
    labels={r['source_image_id']:set(r['labels']) for r in plan['eligibility'] if r['eligible']}
    freeids=set(sum(free,[]))
    extra,targets=assign_splits({i:labels[i] for i in freeids},free,policy['default_ratios'],42) if freeids else ({},{})
    assignments.update(extra)
    assert all(assignments[i]==split for i,split in locked.items()) and set(assignments)==eligible
    split_rows=[]
    for split in ('train','validation','test'):
        oldset={i for i,s in locked.items() if s==split};newset={i for i in newids if assignments[i]==split}
        split_rows.append({'split':split,'original_images':len(oldset),'appended_images':len(newset),
            'proposed_total_images':len(oldset|newset),'paper_images':len(newset),'paper_boxes':sum(paper_by_image[i] for i in newset),
            'original_IDs_sha256':hashlib.sha256(json.dumps(sorted(oldset)).encode()).hexdigest(),'new_image_ids':sorted(newset)})
    assert [v['original_images'] for v in split_rows]==[221,47,47]
    paper_split_support=[]
    for name in names:
        row={'name':name}
        for split in ('train','validation','test'):
            values=[a for a in rawpaper if a['category_id']==by_name[name] and a['image_id'] in kept and assignments[a['image_id']]==split]
            row[split+'_boxes']=len(values);row[split+'_images']=len({a['image_id'] for a in values})
        paper_split_support.append(row)
    train_only={'train_total_images':221+len(newids),'train_paper_boxes':sum(paper_by_image[i] for i in newids),
        'validation_images':47,'validation_paper_boxes':0,'test_images':47,'test_paper_boxes':0}
    counts={'raw_paper_boxes':len(rawpaper),'raw_paper_images':len(paperimages),
        'strict_paper_boxes':sum(paper_by_image[i] for i in kept),'strict_paper_images':len(kept),
        'existing_eligible_images':315,'overlap_paper_with_existing315':len(paperimages&oldids),
        'overlap_strict_paper_with_existing315':len(kept&oldids),'newly_eligible_images':len(newids),
        'proposed_total_eligible_images':len(eligible),
        'proposed_total_annotations':sum(r['annotation_count'] for r in plan['eligibility'] if r['eligible']),
        'original_classes_annotations_on_new_images':sum(len(by_image[i])-paper_by_image[i] for i in newids)}
    result={'authorization':'D-069','scope':'metadata_only_proposal_not_export','counts':counts,'categories':category_rows,
        'waterfall':waterfall,'cooccurrence':cooccurrence,'split_proposal':split_rows,'train_only_alternative':train_only,
        'forced_group_assignments':forced,'new_leakage_group_count':len(free),'free_group_targets':targets,
        'paper_split_support':paper_split_support,'excluded_category_cooccurrence':excluded_categories,
        'proposed_mapping':{'id':8,'name':'paper','source_names':names},
        'evidence':{'raw_current_sha256':before[str(source)],'raw_original_reconstructed_sha256':reconstructed,
            'baseline_eligibility_and_annotation_trace_reproduced_exactly':True,'cached_image_audit':True,
            'exact_groups':plan['exact_duplicate_groups'],'near_candidates':plan['near_duplicate_candidates']},
        'input_sha256':before,'audit_script_sha256':sha(__file__),'photo_reads':0,'photo_decodes':0,
        'test_json_opened':False,'model_imports':False,'export_created':False}
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    assert all(sha(p)==value for p,value in before.items())
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','evidence','split_proposal')},indent=2))
    print('Split counts: '+json.dumps([{k:v for k,v in row.items() if k!='new_image_ids'} for row in split_rows]))
    print('All metadata input hashes unchanged: '+str(len(before)))
    return 0


if __name__=='__main__':raise SystemExit(main())
