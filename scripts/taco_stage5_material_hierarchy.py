"""D-072 CPU-only, reversible saved-query material diagnostics."""
from collections import defaultdict
import copy
from .taco_training_labels_v2 import LABELS
from .taco_diagnostic_nms import diagnostic_nms, NMS_MODES
from .taco_error_analysis import iou

SCHEMES = ('9-specific', '5-material', '4-material')
NAMES = {'9-specific': LABELS,
 '5-material': ('hazardous_cigarette','plastic_waste','metal_waste','styrofoam_waste','paper_waste'),
 '4-material': ('hazardous_cigarette','plastic_polymer_waste','metal_waste','paper_waste')}
MAPS = {'9-specific': tuple(range(9)), '5-material': (0,1,1,1,1,2,1,3,4),
        '4-material': (0,1,1,1,1,2,1,1,3)}
MATERIALS = ('cellulose_acetate_filter','plastic','plastic','plastic','plastic','metal','plastic','polystyrene_foam','paper_fiber')
VIEWS = ('5-Material General (Paper / Plastic / Metal / Styrofoam / Hazardous)',
 '4-Material Polymer (Paper / Plastic+Polymer / Metal / Hazardous)',
 '9-Specific (Subtype Only)', 'Hierarchical (General [Subtype])')
VIEW_SCHEMES = dict(zip(VIEWS, ('5-material','4-material','9-specific','5-material')))
MODES = dict(zip(('raw','class_aware_iou_0.50','class_aware_iou_0.50_containment_0.80'),
                 (NMS_MODES[0],NMS_MODES[2],NMS_MODES[4])))
COLORS = {'hazardous_cigarette':'#e6194B','plastic_waste':'#4363d8',
 'plastic_polymer_waste':'#4363d8','metal_waste':'#911eb4',
 'styrofoam_waste':'#f032e6','paper_waste':'#808000'}


def map_row(row, scheme='5-material', hierarchical=False):
    specific = row.get('specific_category_id',row['category_id'])
    if type(specific) is not int or not 0 <= specific < 9:
        raise ValueError('Invalid specific class')
    category = MAPS[scheme][specific]
    name = NAMES[scheme][category]
    return dict(row, category_id=category, specific_category_id=specific,
                specific_class=LABELS[specific], material=MATERIALS[specific],
                hazard_flag=specific==0, mapped_category=name, taxonomy_scheme=scheme,
                display_label=name + (' ['+LABELS[specific]+']' if hierarchical else ''))


def prepare_rows(rows, scheme='5-material', mode='raw', threshold=0., hierarchical=False):
    mapped = [map_row(r,scheme,hierarchical) for r in rows
              if r.get('valid_clipped_box',True) and r.get('score',1.) >= threshold
              and min(r['bbox'][2:]) > 0]
    mapped.sort(key=lambda r:(-r.get('score',1.),r.get('query_id',r.get('id',0))))
    return diagnostic_nms(mapped, MODES[mode])


def evaluate_case(data, rows, scheme, mode):
    """Global top100 after NMS for COCO; all retained queries for nonexclusive coverage."""
    from .taco_evaluation_v3 import coco_metrics
    mapped_data=copy.deepcopy(data)
    mapped_data['annotations']=[map_row(a,scheme) for a in data['annotations']]
    mapped_data['categories']=[{'id':i,'name':n} for i,n in enumerate(NAMES[scheme])]
    by_image=defaultdict(list)
    for row in rows: by_image[row['image_id']].append(row)
    retained={i['id']:prepare_rows(by_image[i['id']],scheme,mode) for i in data['images']}
    predictions=[{k:r[k] for k in ('image_id','category_id','score','bbox')}
                 for rs in retained.values() for r in rs[:100]]
    metrics=coco_metrics(mapped_data,predictions,cat_ids=list(range(len(NAMES[scheme]))),label_names=NAMES[scheme])
    coverage=[{'annotation_id':a['id'],'image_id':a['image_id'],
               'covered':any(r['score']>=.05 and r['category_id']==a['category_id']
                   and iou(r['bbox'],a['bbox'])>=.5 for r in retained[a['image_id']])}
              for a in mapped_data['annotations']]
    return {'metrics':metrics,'covered':sum(a['covered'] for a in coverage),
            'objects':len(coverage),'coverage':coverage,
            'retained_queries':sum(map(len,retained.values()))}


def main():
    import hashlib,json,uuid,csv
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    run1=root/'runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4'
    sources={'baseline_256_384':run1/'baseline.json',
     'epoch2_256_384':run1/'validation-epoch-2.json',
     'blend_2575_384_576':run1/'eval_blend_2575_384_576.json',
     'highres_epoch1_384_576':root/'runs/training/stage5-paper-558bcc18d6cf4556a966d8cfc320c968/validation-epoch-1.json'}
    val=root/'exports/taco/strict-clean-stage5-v1/validation.json'
    legacy=root/'exports/taco/strict-clean-stage2-v1/validation.json'
    protected=[root/'scripts'/n for n in ('train_taco.py','taco_training_data.py','taco_evaluation.py','training_checkpoint.py','training_adapter.py','training_smoke.py')]+[root/'config/taco_taxonomy.v2.json']
    inputs=list(sources.values())+[val,legacy]+protected
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p.relative_to(root)):digest(p) for p in inputs}
    data=json.loads(val.read_text());old={i['id'] for i in json.loads(legacy.read_text())['images']}
    assert len(data['images'])==65 and len(old)==47
    out=root/'runs/training'/('stage5-material-hierarchy-'+uuid.uuid4().hex)
    out.mkdir()
    result=[]
    for state,path in sources.items():
        saved=json.loads(path.read_text());rows=saved['raw_queries']
        ids={i['id'] for i in data['images']}
        assert len(rows)==65*900 and {r['image_id'] for r in rows}==ids
        assert len({(r['image_id'],r['query_id']) for r in rows})==len(rows)
        for view,subset in [('combined_65',ids),('legacy_47',old),('annex_18',ids-old)]:
            sub=dict(data,images=[i for i in data['images'] if i['id'] in subset],annotations=[a for a in data['annotations'] if a['image_id'] in subset])
            for scheme in SCHEMES:
                for mode in MODES:
                    case=evaluate_case(sub,[r for r in rows if r['image_id'] in subset],scheme,mode)
                    case.update(state=state,view=view,taxonomy=scheme,nms=mode)
                    result.append(case)
        print(state, 'complete', flush=True)
    assert before=={str(p.relative_to(root)):digest(p) for p in inputs}
    (out/'summary.json').write_text(json.dumps({'diagnostic_only':True,'legacy_conditioning':'nine-class prompt subset; not original eight-class prompt evaluation','input_sha256':before,'cases':result},indent=2))
    fields=['state','view','taxonomy','nms','AP','AP50','AP75','AR100','covered','objects']
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for c in result:writer.writerow({k:c.get(k,c['metrics'].get(k)) for k in fields})
    print(out)

if __name__=='__main__':main()
