"""Annotation-relative diagnostics over saved results; never visual truth or AP."""
from collections import Counter
import math

from .training_smoke import LABELS

MODELS = ('pretrained','epoch-01','epoch-02')


def iou(a,b):
    x,y,w,h=a;u,v,s,t=b
    intersection=max(0,min(x+w,u+s)-max(x,u))*max(0,min(y+h,v+t)-max(y,v))
    union=w*h+s*t-intersection
    return intersection/union if union>0 else 0.


def saved_rows(data, evaluation, *, queries=900):
    """Validate identity/coordinates and reconstruct official membership exactly."""
    frames={i['id']:i for i in data['images']}
    rows={key:[] for key in frames}
    for row in evaluation['raw_queries']:
        if row['image_id'] not in frames:
            raise ValueError('Saved prediction belongs to another image/corpus')
        image=frames[row['image_id']];box=row['bbox']
        if (type(row['query_id']) is not int or not 0<=row['query_id']<queries
                or type(row['category_id']) is not int or not 0<=row['category_id']<8
                or not isinstance(row['raw_phrase'],str)
                or not isinstance(row['score'],(int,float)) or not math.isfinite(row['score'])
                or not 0<=row['score']<=1 or len(box)!=4
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in box)):
            raise ValueError('Invalid saved query identity/score/coordinates')
        x,y,w,h=box
        if (min(x,y,w,h)<0 or x+w>image['width']+1e-5 or y+h>image['height']+1e-5
                or row['valid_clipped_box'] is not (w>0 and h>0)):
            raise ValueError('Saved box is outside original-image frame')
        rows[row['image_id']].append(dict(row))
    expected=[]
    for image in data['images']:
        group=rows[image['id']]
        if len(group)!=queries or {r['query_id'] for r in group}!=set(range(queries)):
            raise ValueError('Missing/duplicate saved query IDs')
        if [r['query_id'] for r in group]!=[r['query_id'] for r in sorted(group,key=lambda r:(-r['score'],r['query_id']))]:
            raise ValueError('Saved query ranking differs from official stable ranking')
        kept=0
        for rank,row in enumerate(group,1):
            row['rank']=rank
            row['retained']=row['valid_clipped_box'] and kept<100
            if row['retained']:
                expected.append({k:row[k] for k in ('image_id','category_id','score','bbox')})
                kept+=1
    if evaluation['predictions']!=expected:
        raise ValueError('Saved official predictions do not match raw query identities')
    return rows


def diagnose(annotations, rows, threshold=.05):
    """Greedy score-ranked class-aware IoU .50; NOT COCO's full matching protocol.

    Prediction categories are annotation-relative, not verified false positives.
    Candidate coverage is non-exclusive and separately identifies rank/threshold
    suppression; multiple annotations may share a candidate without forced counts.
    """
    retained=[r for r in rows if r['retained'] and r['valid_clipped_box']]
    all_valid=[r for r in rows if r['valid_clipped_box']]
    matched=set();prediction_errors=[];prediction_counts=Counter()
    for row in retained:
        if row['score']<threshold:continue
        overlaps=[(iou(row['bbox'],a['bbox']),a) for a in annotations]
        available=[(value,a) for value,a in overlaps if a['category_id']==row['category_id'] and a['id'] not in matched]
        value,ann=max(available,key=lambda pair:(pair[0],-pair[1]['id']),default=(0,None))
        if value>=.5:
            matched.add(ann['id']);kind='matched'
        else:
            prior=[(v,a) for v,a in overlaps if v>=.5 and a['id'] in matched and a['category_id']==row['category_id']]
            value,ann=max(overlaps,key=lambda pair:(pair[0],-pair[1]['id']),default=(0,None))
            kind=('duplicate' if prior else 'wrong_class' if value>=.5
                  else 'localization_overlap_010_to_050' if value>=.1 else 'unmatched_prediction')
        prediction_counts[kind]+=1
        prediction_errors.append({'query_id':row['query_id'],'category_id':row['category_id'],
            'kind':kind,'best_annotation_id':ann['id'] if ann else None,'iou':value,
            'score':row['score']})
    annotation_rows=[]
    for ann in annotations:
        same=[r for r in all_valid if r['category_id']==ann['category_id']]
        same_kept=[r for r in retained if r['category_id']==ann['category_id']]
        best_same=max((iou(r['bbox'],ann['bbox']) for r in same),default=0.)
        best_kept=max((iou(r['bbox'],ann['bbox']) for r in same_kept),default=0.)
        same_compatible=[r for r in same_kept if iou(r['bbox'],ann['bbox'])>=.5]
        best_any=max((iou(r['bbox'],ann['bbox']) for r in all_valid),default=0.)
        kind=('matched' if ann['id'] in matched else
              'assignment_conflict' if any(r['score']>=threshold for r in same_compatible) else
              'threshold_hidden' if best_kept>=.5 else
              'outside_top100' if best_same>=.5 else
              'wrong_class_only' if best_any>=.5 else
              'poor_localization_candidate' if best_any>=.1 else 'no_substantial_overlap')
        annotation_rows.append({'annotation_id':ann['id'],'category_id':ann['category_id'],
            'kind':kind,'best_same_class_iou_all900':best_same,
            'best_same_class_iou_top100':best_kept,'best_any_class_iou_all900':best_any,
            'best_compatible_score_top100':max((r['score'] for r in same_compatible),default=None)})
    return {'threshold':threshold,'iou_threshold':.5,'prediction_counts':dict(prediction_counts),
        'annotation_counts':dict(Counter(r['kind'] for r in annotation_rows)),
        'annotations':annotation_rows,'predictions':prediction_errors}


def analyze(data, models, thresholds=(0.,.05,.10,.25)):
    images=[]
    for image in data['images']:
        anns=[a for a in data['annotations'] if a['image_id']==image['id']]
        images.append({'image_id':image['id'],'models':{name:{str(t):diagnose(anns,models[name][image['id']],t)
            for t in thresholds} for name in MODELS}})
    aggregate={}
    for name in MODELS:
        aggregate[name]={}
        for threshold in thresholds:
            key=str(threshold);prediction_counts=Counter();annotation_counts=Counter()
            classes={label:Counter() for label in LABELS}
            for image in images:
                diagnostic=image['models'][name][key]
                prediction_counts.update(diagnostic['prediction_counts'])
                annotation_counts.update(diagnostic['annotation_counts'])
                for ann in diagnostic['annotations']:classes[LABELS[ann['category_id']]][ann['kind']]+=1
            aggregate[name][key]={'annotation_counts':dict(annotation_counts),
                'prediction_counts':dict(prediction_counts),'classes':{k:dict(v) for k,v in classes.items()}}
    return {'method':'annotation_relative_greedy_IoU050_not_visual_confirmation_not_COCO_AP',
            'thresholds':list(thresholds),'images':images,'aggregate':aggregate}


def representative_sample(data, analysis, limit=16):
    if limit<8:raise ValueError('Representative limit must permit all eight classes')
    images={i['id']:i for i in data['images']};selected={}
    def add(image_id,reason):
        if image_id in selected or len(selected)<limit:selected.setdefault(image_id,[]).append(reason)
    def fraction(ann):
        image=images[ann['image_id']]
        return ann['bbox'][2]*ann['bbox'][3]/(image['width']*image['height'])
    for category,label in enumerate(LABELS):
        anns=sorted([a for a in data['annotations'] if a['category_id']==category],key=lambda a:(fraction(a),a['image_id'],a['id']))
        if anns:add(anns[0]['image_id'],'smallest normalized bbox for '+label)
    candidates=analysis['images']
    def pick(reason,key):
        ordered=sorted(candidates,key=lambda row:(-key(row),row['image_id']))
        if ordered and key(ordered[0])>0:add(ordered[0]['image_id'],reason)
    def matches(row,name):return row['models'][name]['0.0']['annotation_counts'].get('matched',0)
    counts=Counter(a['image_id'] for a in data['annotations'])
    pick('most annotated objects',lambda r:counts[r['image_id']] if counts[r['image_id']]>1 else 0)
    pick('pretrained beats epoch1 at score floor zero',lambda r:matches(r,'pretrained')-matches(r,'epoch-01'))
    pick('epoch1 improves over pretrained at score floor zero',lambda r:matches(r,'epoch-01')-matches(r,'pretrained'))
    pick('epoch2 deteriorates versus epoch1 at score floor zero',lambda r:matches(r,'epoch-01')-matches(r,'epoch-02'))
    pick('most threshold-hidden annotations at .25',lambda r:max(r['models'][n]['0.25']['annotation_counts'].get('threshold_hidden',0) for n in MODELS))
    for kind in ('wrong_class','localization_overlap_010_to_050','duplicate','unmatched_prediction'):
        pick('largest '+kind+' count across models at diagnostic .05',lambda r,k=kind:max(
            r['models'][n]['0.05']['prediction_counts'].get(k,0) for n in MODELS))
    for category,label in enumerate(LABELS):
        anns=sorted([a for a in data['annotations'] if a['category_id']==category],key=lambda a:(fraction(a),a['image_id'],a['id']))
        if anns:add(anns[len(anns)//2]['image_id'],'median normalized bbox for '+label)
    return [{'image_id':key,'reasons':value,'human_visual_confirmation':'pending'} for key,value in selected.items()]
