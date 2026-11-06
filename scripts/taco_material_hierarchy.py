"""Reversible saved-prediction diagnostics; never an official taxonomy change."""
from .training_smoke import LABELS

TAXONOMY_VIEWS = ('8 specific classes (official)',
                  '4 material/hazard categories (mapped)',
                  '3 material/hazard categories (polymer grouped)')
SCHEMES = ('8-specific', '4-material', '3-material')
MAPS = {'8-specific': tuple(range(8)), '4-material': (0,1,1,1,1,2,1,3),
        '3-material': (0,1,1,1,1,2,1,1)}
NAMES = {'8-specific': LABELS,
         '4-material': ('hazardous_cigarette','plastic_waste','metal_waste','styrofoam_waste'),
         '3-material': ('hazardous_cigarette','plastic_polymer_waste','metal_waste')}
MATERIALS = ('cellulose_acetate_filter','plastic','plastic','plastic','plastic','metal','plastic','polystyrene_foam')


def scheme_key(value):
    if value in TAXONOMY_VIEWS:return SCHEMES[TAXONOMY_VIEWS.index(value)]
    if value not in SCHEMES:raise ValueError('Unknown diagnostic taxonomy view')
    return value


def map_row(row, scheme='8-specific'):
    """Copy a row, preserving its specific ID/name, geometry, score and provenance."""
    scheme=scheme_key(scheme)
    specific=row.get('specific_category_id',row['category_id'])
    if type(specific) is not int or not 0<=specific<8:raise ValueError('Invalid specific category ID')
    if scheme=='8-specific' and 'specific_category_id' not in row:return dict(row)
    mapped=dict(row,category_id=MAPS[scheme][specific],specific_category_id=specific,
                specific_class=LABELS[specific],hazard_flag=specific==0,
                material=MATERIALS[specific],taxonomy_scheme=scheme)
    mapped['mapped_category']=NAMES[scheme][mapped['category_id']]
    mapped['display_label']=mapped['mapped_category']+' ['+mapped['specific_class']+']'
    return mapped


def row_label(row):
    return row.get('display_label',LABELS[row['category_id']])


def mapped_coco_metrics(data, predictions, scheme):
    """CPU COCO on remapped GT/predictions, separate from official evaluator.

    Original-pixel areas and COCO .50:.05:.95, maxDets 1/10/100 are retained.
    Caller applies any global top100 cap after remapping and diagnostic NMS.
    """
    import copy,contextlib,io
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    scheme=scheme_key(scheme)
    mapped=copy.deepcopy(data)
    mapped['annotations']=[map_row(a,scheme) for a in data['annotations']]
    mapped['categories']=[{'id':i,'name':n} for i,n in enumerate(NAMES[scheme])]
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=mapped;gt.createIndex()
        if predictions:dt=gt.loadRes(copy.deepcopy(predictions))
        else:
            dt=COCO();dt.dataset={'images':mapped['images'],'categories':mapped['categories'],'annotations':[]};dt.createIndex()
        ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=[i['id'] for i in data['images']]
        ev.params.catIds=list(range(len(NAMES[scheme])))
        ev.evaluate();ev.accumulate();ev.summarize()
    def finite(v):return float(v) if np.isfinite(v) and v>=0 else None
    keys=('AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large')
    result=dict(zip(keys,map(finite,ev.stats)));result['classes']={}
    for k,name in enumerate(NAMES[scheme]):
        precision=ev.eval['precision'][:,:,k,0,2];recall=ev.eval['recall'][:,k,0,2]
        objects=[a for a in mapped['annotations'] if a['category_id']==k]
        result['classes'][name]={'objects':len(objects),'images':len({a['image_id'] for a in objects}),
             'AP':finite(precision[precision>=0].mean()) if (precision>=0).any() else None,
             'AR100':finite(recall[recall>=0].mean()) if (recall>=0).any() else None}
    return result


def greedy_matching(annotations, rows, threshold=.05):
    """One prediction/GT at most once, score desc/query ID asc, IoU >= .50.

    Caller supplies one image, active taxonomy and optional post-remap NMS rows.
    GT ties use ascending annotation ID. No COCO maxDets cap is imposed here.
    """
    from .taco_error_analysis import iou
    matched=set();assignments=[];fp=0
    for row in sorted(rows,key=lambda r:(-r['score'],r['query_id'])):
        if not row['valid_clipped_box'] or row['score']<threshold:continue
        value,ann=max(((iou(row['bbox'],a['bbox']),a) for a in annotations
                       if a['id'] not in matched and a['category_id']==row['category_id']),
                      key=lambda pair:(pair[0],-pair[1]['id']),default=(0,None))
        if value>=.5:
            matched.add(ann['id']);assignments.append({'query_id':row['query_id'],'annotation_id':ann['id'],'iou':value})
        else:fp+=1
    tp=len(matched);fn=len(annotations)-tp
    p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
    return {'TP':tp,'FP':fp,'FN':fn,'Precision':p,'Recall':r,'F1':2*p*r/(p+r) if p+r else 0.,
            'assignments':assignments}
