"""D-074 immutable Tier1 export; reconstruct held-out JSON from metadata only."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.configuration import strict_load
from scripts.taco_dataset import image_path, sha256
from scripts.taco_training_data_v2 import dataset_context


def serialized(data):
    # Match original Windows write_text serialization, including CRLF.
    return (json.dumps(data,indent=2,allow_nan=False,ensure_ascii=False,sort_keys=True)+'\n').replace('\n','\r\n').encode('utf-8')


def reconstruct_frozen(raw, ledger, template, taxonomy):
    """Replay original global annotation numbering without reading any split JSON."""
    mapping={n:label['id'] for label in taxonomy['labels'] for n in label['taco_categories']}
    categories={c['id']:c for c in raw['categories']}
    eligible={r['source_image_id']:r for r in ledger if r['eligible']}
    exports={s:{k:copy.deepcopy(template[k]) for k in ('info','licenses','categories')} for s in ('train','validation','test')}
    for d in exports.values():d.update(images=[],annotations=[])
    for r in ledger:
        if not r['eligible']:continue
        original=r['source_image'];image={k:original[k] for k in ('id','file_name','width','height')}
        image['source_taco']={'image':copy.deepcopy(original),'image_record_index':r['image_record_index'],
          'file_sha256':r['fingerprint']['file_sha256'],'pixel_sha256':r['fingerprint']['pixel_sha256']}
        exports[r['split']]['images'].append(image)
    next_id=1
    for index,a in enumerate(raw['annotations']):
        if a['image_id'] not in eligible:continue
        category=categories[a['category_id']]
        row=copy.deepcopy(a);row.update(id=next_id,category_id=mapping[category['name']],
          source_taco={'annotation_id':a['id'],'category':category,'annotation_record_index':index,'annotation':copy.deepcopy(a)})
        exports[eligible[a['image_id']]['split']]['annotations'].append(row);next_id+=1
    return exports,next_id


def main():
    old=ROOT/'exports/taco/strict-clean-stage5-v1'
    target=ROOT/'exports/taco/strict-clean-stage5-tier1-v1'
    rights_path=ROOT/'work/stage5-paper/rights_tier1.json'
    if target.exists() or rights_path.exists():raise FileExistsError('Tier1 artifacts already exist; never overwrite')
    audit_path=ROOT/'runs/training/stage5-data-expansion-audit-5412a13c0258467ba33a1f475cc4b65c/summary.json'
    audit=strict_load(audit_path);manifest=strict_load(old/'preparation_manifest.json')
    for p,h in audit['input_sha256'].items():
        assert sha256(p)==h,p
    source=Path(manifest['source']['annotations']);raw=strict_load(source)
    taxonomy=strict_load(ROOT/'config/taco_taxonomy.v2.json')
    ledger=[json.loads(s) for s in (old/'image_eligibility.jsonl').read_text().splitlines()]
    validation=strict_load(old/'validation.json');validation_bytes=(old/'validation.json').read_bytes()
    original,next_id=reconstruct_frozen(raw,ledger,validation,taxonomy)
    assert serialized(original['train'])==(old/'train.json').read_bytes(),'Original train reconstruction differs'
    assert serialized(original['validation'])==validation_bytes,'Original validation reconstruction differs'
    frozen_test_bytes=serialized(original['test'])
    assert hashlib.sha256(frozen_test_bytes).hexdigest()==manifest['artifacts']['test.json'],'Frozen test reconstruction hash differs'
    trainids=set(audit['tiers']['tier1']['training_image_ids']);locked=set(audit['frozen']['test']['ids'])|set(audit['frozen']['validation']['ids'])
    assert len(trainids)==336 and not trainids&locked and 1173 not in trainids
    evidence={r['id']:r for r in audit['image_evidence']}
    rawimages={i['id']:i for i in raw['images']};dataset=Path(manifest['source']['dataset_root']).resolve()
    permitted={image_path(dataset,rawimages[i]['file_name']) for i in trainids|set(audit['frozen']['validation']['ids'])}
    tests={image_path(dataset,rawimages[i]['file_name']) for i in audit['frozen']['test']['ids']}
    # At runtime only the new export and versioned rights record may be created.
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve();flags=args[2] if len(args)>2 else 0
            writing=isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
            if path in tests:raise AssertionError('Test photo access prohibited')
            if path.name=='test.json' and not (writing and path==target/'test.json'):
                raise AssertionError('Test JSON reads prohibited')
            if writing and not (path.is_relative_to(target) or path==rights_path):raise AssertionError('Existing artifacts are immutable')
            if path.is_relative_to(dataset) and path.suffix.lower() in ('.jpg','.jpeg','.png') and path not in permitted:
                raise AssertionError('Photo outside approved 401 corpus')
    sys.addaudithook(guard)
    target.mkdir()
    train={k:copy.deepcopy(validation[k]) for k in ('info','licenses','categories')};train.update(images=[],annotations=[])
    indices={r['id']:idx for idx,r in enumerate(raw['images'])}
    for i in raw['images']:
        if i['id'] not in trainids:continue
        fp=evidence[i['id']]['fingerprint'];image={k:i[k] for k in ('id','file_name','width','height')}
        image['source_taco']={'image':copy.deepcopy(i),'image_record_index':indices[i['id']],
             'file_sha256':fp['file_sha256'],'pixel_sha256':fp['pixel_sha256']}
        train['images'].append(image)
    mapping={n:l['id'] for l in taxonomy['labels'] for n in l['taco_categories']};categories={c['id']:c for c in raw['categories']}
    for index,a in enumerate(raw['annotations']):
        if a['image_id'] not in trainids:continue
        category=categories[a['category_id']];row=copy.deepcopy(a)
        row.update(id=next_id,category_id=mapping[category['name']],source_taco={'annotation_id':a['id'],
            'category':category,'annotation_record_index':index,'annotation':copy.deepcopy(a)})
        train['annotations'].append(row);next_id+=1
    assert len(train['annotations'])==637
    photos={}
    for split,data in [('train',train),('validation',validation)]:
        for image in data['images']:
            path=image_path(dataset,image['file_name']);payload=path.read_bytes();h=hashlib.sha256(payload).hexdigest()
            assert h==image['source_taco']['file_sha256']
            with Image.open(io.BytesIO(payload)) as im:
                im.load();assert im.size==(image['width'],image['height']) and im.getexif().get(274,1)==1
            photos[str(image['id'])]={'split':split,'file_name':image['file_name'],'sha256':h}
    assert len(photos)==401
    artifacts={}
    def write(name,payload):
        with (target/name).open('xb') as f:f.write(payload)
        artifacts[name]=hashlib.sha256(payload).hexdigest()
    write('train.json',serialized(train));write('validation.json',validation_bytes);write('test.json',frozen_test_bytes)
    write('taxonomy_snapshot.json',serialized(taxonomy));write('policy_snapshot.json',(old/'policy_snapshot.json').read_bytes())
    # Ledger carries original frozen references, fresh training selection and duplicate evidence.
    assignments={i:'train' for i in trainids}
    for split in ('validation','test'):assignments.update({i:split for i in audit['frozen'][split]['ids']})
    newledger=[]
    for i in raw['images']:
        ident=i['id'];newledger.append({'source_image_id':ident,'source_image':i,'eligible':ident in assignments,
             'split':assignments.get(ident),'fingerprint':evidence[ident]['fingerprint']})
    write('image_eligibility.jsonl',(''.join(json.dumps(r,sort_keys=True)+'\n' for r in newledger)).encode())
    write('duplicate_review.json',serialized(audit['duplicates']))
    write('photo-verification.json',serialized(photos))
    newmanifest=copy.deepcopy(manifest);newmanifest.update(authorization='D-074',input_audit=str(audit_path),
      artifacts=artifacts,split_algorithm='D073_tier1_train_with_frozen_stage5_65_65',
      source=dict(manifest['source'],sha256=sha256(source)),
      frozen_test_reconstructed_from_metadata=True,frozen_test_json_sha256=artifacts['test.json'],
      tier1={'train_images':336,'train_annotations':637,'validation_images':65,'validation_annotations':109,'test_images':65})
    write('preparation_manifest.json',serialized(newmanifest))
    record=strict_load(ROOT/'work/stage5-paper/rights.json')
    record.update(approving_authority='Project owner via D-074 chat; no third-party authority claimed',approval_date='2026-10-29',
      dataset=dataset_context(target),photo_sha256=photos,
      assessment={'path':'docs/TACO_STAGE_5_ACADEMIC_USE.md','sha256':sha256(ROOT/'docs/TACO_STAGE_5_ACADEMIC_USE.md')})
    record['factors']['amount']='Exact 336 train / 65 validation photographs and annotations, 401 unique photos; whole-photo local spatial research with uncertainty; 65 test images unused.'
    record['authorization']='D-074';record['limits']={'training_updates':336,'validation_forwards':336,'test_access':False}
    with rights_path.open('xb') as f:f.write(serialized(record))
    print(json.dumps({'export':str(target),'rights':str(rights_path),'verified_photos':len(photos),'frozen_test_hash_matches_original':True}))

if __name__=='__main__':main()
