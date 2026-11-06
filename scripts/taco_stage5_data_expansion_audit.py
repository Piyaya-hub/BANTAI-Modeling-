"""D-073 proposal only: fresh non-held-out raster audit, frozen metadata references."""
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.taco_dataset import audit_dataset, image_path, sha256
from scripts.taco_preparation import make_plan, load_policy, annotation_issues, RULES
from scripts.taco_splits import duplicate_groups, fingerprint, Groups

SIBLINGS = {2:('Polypropylene bag',),3:('Six pack rings',),
            5:('Food Can','Aerosol'),6:('Disposable food container','Spread tub','Other plastic container'),
            7:('Foam cup','Foam food container'),8:('Plastified paper bag',)}
ACCESSORIES = frozenset(('Plastic bottle cap','Metal bottle cap','Plastic lid','Metal lid',
                        'Plastic straw','Plastic utensils','Pop tab','Aluminium foil','Glass bottle'))


def proposal_audit(audit, siblings=False, filter_accessories=False, other_plastic=False):
    """Copy metadata; only valid, non-crowd accessory annotations may be omitted."""
    result=copy.deepcopy(audit)
    names={n:label for label in result['taxonomy']['labels'] for n in label['taco_categories']}
    if siblings:
        for cid,extra in SIBLINGS.items():
            result['taxonomy']['labels'][cid]['taco_categories'].extend(extra)
        if other_plastic:result['taxonomy']['labels'][3]['taco_categories'].append('Other plastic')
        names={n:label for label in result['taxonomy']['labels'] for n in label['taco_categories']}
    kept=[];omitted=[]
    for row in result['annotations']:
        name=row['original_category']['name'] if row['original_category'] else None
        if name in names:
            label=names[name];row.update(bantai_label=label['name'],bantai_label_id=label['id'],handling='mapped')
        safe_accessory=(name in ACCESSORIES and not annotation_issues(row)
                        and row['original'].get('iscrowd')==0 and row['original'].get('ignore',0)==0)
        if filter_accessories and safe_accessory:omitted.append(row['original']['id'])
        else:kept.append(row)
    result['annotations']=kept
    return result,omitted


def locked_duplicate_members(fingerprints, locked, policy):
    """Conservatively block entire transitive components, including ineligible bridges."""
    exact,near=duplicate_groups(fingerprints,policy)
    groups=Groups(fingerprints)
    for group in exact:
        for ident in group[1:]:groups.join(group[0],ident)
    for pair in near:groups.join(pair['image_id_a'],pair['image_id_b'])
    touching=[g for g in groups.members() if set(g)&locked]
    return set(i for g in touching for i in g),{'exact_groups':exact,'near_pairs':near,'frozen_touching_components':touching}


def summarize(plan, audit, locked, blocked, baseline, omitted):
    eligible={r['source_image_id'] for r in plan['eligibility'] if r['eligible']}
    selected=eligible-locked-blocked
    annotations=[r for r in audit['annotations'] if r['original']['image_id'] in selected]
    classes=[]
    for cid in range(9):
        rows=[a for a in annotations if a['bantai_label_id']==cid]
        classes.append({'id':cid,'name':audit['taxonomy']['labels'][cid]['name'],'boxes':len(rows),
                        'images':len({a['original']['image_id'] for a in rows})})
    reasons=Counter()
    for row in plan['eligibility']:
        ident=row['source_image_id']
        if ident in locked:reasons['frozen_heldout']+=1
        elif not row['eligible']:reasons[row['primary_reason']]+=1
        elif ident in blocked:reasons['duplicate_component_touches_heldout']+=1
        else:reasons['training_candidate']+=1
    return {'training_images':len(selected),'boxes':len(annotations),'classes':classes,
            'training_image_ids':sorted(selected),'new_image_ids':sorted(selected-baseline),
            'baseline_images_lost':sorted(baseline-selected),'new_images':len(selected-baseline),
            'gain_images_vs_baseline':len(selected)-len(baseline),
            'gain_boxes_vs_baseline':len(annotations)-595,
            'excluded_by_primary_reason':dict(reasons),'omitted_accessory_annotation_ids':omitted,
            'omitted_accessories_on_selected_images':sum(r['original']['id'] in omitted and r['original']['image_id'] in selected for r in audit.get('all_annotations',[])),
            'eligibility':[{'id':r['source_image_id'],'eligible':r['eligible'],'reasons':r['reasons']} for r in plan['eligibility']]}


def main():
    from scripts.configuration import strict_load
    stage=ROOT/'exports/taco/strict-clean-stage5-v1'
    review=ROOT/'reports/taco/local-review'
    manifest=strict_load(stage/'preparation_manifest.json')
    dataset=Path(manifest['source']['dataset_root']).resolve()
    source=Path(manifest['source']['annotations'])
    raw=strict_load(source)
    ledger=[json.loads(s) for s in (stage/'image_eligibility.jsonl').read_text().splitlines()]
    cached=[json.loads(s) for s in (review/'images.jsonl').read_text().splitlines()]
    frozen={split:{r['source_image_id'] for r in ledger if r['eligible'] and r['split']==split}
            for split in ('validation','test')}
    baseline={r['source_image_id'] for r in ledger if r['eligible'] and r['split']=='train'}
    locked=frozen['validation']|frozen['test']
    assert len(baseline)==307 and len(locked)==130 and not baseline&locked
    assert all(len(v)==65 for v in frozen.values()) and len(raw['images'])==1500
    by_id={r['original']['id']:r for r in cached}
    oldfp={r['source_image_id']:r['fingerprint'] for r in ledger if r['fingerprint']}
    assert all(by_id[i]['sha256']==oldfp[i]['file_sha256'] and by_id[i]['original']==next(x for x in raw['images'] if x['id']==i) for i in locked)
    out=ROOT/'runs/training'/('stage5-data-expansion-audit-'+uuid.uuid4().hex)
    out.mkdir()
    heldout_paths={image_path(dataset,i['file_name']) for i in raw['images'] if i['id'] in locked}
    inputs=[source,stage/'preparation_manifest.json',stage/'image_eligibility.jsonl',stage/'train.json',stage/'validation.json',
            review/'images.jsonl',ROOT/'config/taco_taxonomy.v2.json',ROOT/'config/taco_preparation_policy.v1.json']
    inputs += [ROOT/'scripts'/n for n in ('train_taco.py','taco_training_data.py','taco_evaluation.py','training_checkpoint.py','training_adapter.py','training_smoke.py')]
    before={str(p):sha256(p) for p in inputs}
    for p in (stage/'train.json',stage/'validation.json',stage/'image_eligibility.jsonl'):
        assert sha256(p)==manifest['artifacts'][p.name]
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.name=='test.json' or p in heldout_paths or p.suffix.lower() in ('.pt','.bin','.safetensors'):
                raise AssertionError('Protected test/heldout/checkpoint read: '+str(p))
            flags=args[2] if len(args)>2 else 0
            if isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
                if not p.is_relative_to(out):raise AssertionError('Write outside new audit folder: '+str(p))
        elif event in ('os.remove','os.rename','os.rmdir','os.mkdir'):
            values=args[:2] if event=='os.rename' else args[:1]
            for value in values:
                if isinstance(value,(str,bytes,os.PathLike)) and not Path(os.fsdecode(value)).resolve().is_relative_to(out):
                    raise AssertionError('Mutation outside audit directory')
    sys.addaudithook(guard)
    names={i['file_name']:i['id'] for i in raw['images']}
    def permitted_path(root,name):
        return out/('heldout-not-opened-'+str(names[name])) if names[name] in locked else image_path(root,name)
    print('Fresh physical scan: 1370 non-held-out images; 130 frozen metadata references',flush=True)
    with patch('scripts.taco_dataset.image_path',side_effect=permitted_path):
        audit=audit_dataset(dataset,source,ROOT/'config/taco_taxonomy.v2.json')
    for index,row in enumerate(audit['images']):
        if row['original']['id'] in locked:audit['images'][index]=copy.deepcopy(by_id[row['original']['id']])
    fingerprints={}
    for n,row in enumerate(audit['images'],1):
        ident=row['original']['id']
        if row['status']=='readable':fingerprints[ident]=copy.deepcopy(oldfp[ident]) if ident in locked else fingerprint(row)
        if n%100==0:print('Fingerprints',n,'/ 1500',flush=True)
    # annotation validity is source/declared-box based; restore frozen cached image-status metadata only.
    images={r['original']['id']:r for r in audit['images']}
    for a in audit['annotations']:
        image=images[a['original']['image_id']]
        a['image_status']=image['status'];a['image_issues']=image['issues']
    policy=load_policy();blocked,duplicates=locked_duplicate_members(fingerprints,locked,policy['near_duplicates'])
    tiers={}
    for name,siblings,accessories,other in [('tier1',False,False,False),('tier2',True,False,False),
                    ('tier3',True,True,False),('tier2_with_other_plastic',True,False,True),('tier3_with_other_plastic',True,True,True)]:
        proposed,omitted=proposal_audit(audit,siblings,accessories,other)
        with patch('scripts.taco_preparation.fingerprint',side_effect=lambda row:copy.deepcopy(fingerprints[row['original']['id']])):
            plan=make_plan(proposed,policy)
        proposed['all_annotations']=audit['annotations']
        tiers[name]=summarize(plan,proposed,locked,blocked,baseline,omitted)
        print(name,tiers[name]['training_images'],tiers[name]['boxes'],flush=True)
    base=strict_load(stage/'train.json')
    assert len(base['images'])==307 and len(base['annotations'])==595
    baseclasses=[{'id':i,'name':audit['taxonomy']['labels'][i]['name'],
        'boxes':sum(a['category_id']==i for a in base['annotations']),
        'images':len({a['image_id'] for a in base['annotations'] if a['category_id']==i})} for i in range(9)]
    missing={r['original']['id'] for r in cached if r['status']=='missing'}
    restored={i for i in missing if images[i]['status']=='readable'}
    for tier in tiers.values():
        tier['historically_missing_new_image_ids']=sorted(set(tier['new_image_ids'])&missing)
        tier['historically_missing_new_images']=len(tier['historically_missing_new_image_ids'])
        assert not set(tier['training_image_ids'])&locked
    assert all(sha256(p)==before[str(p)] for p in inputs)
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    result={'authorization':'D-073','proposal_only':True,'dataset':str(dataset),'raw_images':1500,
       'physical_scan_images':1370,'heldout_photos_opened':0,'test_json_opened':False,
       'frozen':{s:{'count':len(ids),'ids':sorted(ids),'ids_sha256':hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()} for s,ids in frozen.items()},
       'baseline':{'training_images':307,'boxes':595,'classes':baseclasses},'tiers':tiers,
       'historical_missing':{'count':len(missing),'now_readable':len(restored),'still_unreadable':sorted(missing-restored)},
       'physical_status_counts':dict(Counter(r['status'] for i,r in images.items() if i not in locked)),
       'image_evidence':[{'id':i,'scope':'cached_frozen' if i in locked else 'fresh','status':r['status'],'issues':r['issues'],
                         'fingerprint':fingerprints.get(i)} for i,r in sorted(images.items())],
       'duplicates':duplicates,'blocked_nonheldout_duplicate_ids':sorted(blocked-locked),
       'mapping_proposals':{'siblings':SIBLINGS,'other_plastic_separate':True,'allowed_omitted_accessories':sorted(ACCESSORIES)},
       'input_sha256':before,'script_sha256':sha256(__file__),'inputs_unchanged':True,
       'limitations':['Frozen 130 photos not accessed; duplicate checks use saved fingerprints for them.',
         'Perceptual hash candidate groups are conservative, not proof that all semantic near-duplicates are detected.',
         'Tier3 omission can teach accessories as background; it is not ignore-region training.',
         'Sibling categories change subtype semantics; mapping compatibility and expanded-corpus lawful-use basis require owner review.']}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(out,flush=True)

if __name__=='__main__':main()
