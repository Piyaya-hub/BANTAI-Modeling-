"""Fixed eight-class span decoder and COCO bbox validation; no test access."""
import contextlib
import copy
import io
import time

from .training_smoke import LABELS, move_batch, checked_loss, validate_predictions
from .taco_training_data import get_batch

PROTOCOL = {'version': 'bantai.taco-evaluation.v1', 'class_score': 'mean_ordinary_subword_sigmoid',
            'query_class': 'single_argmax', 'score_filter': 0.0, 'max_predictions_per_image': 100,
            'nms': False, 'iou': 'COCO_0.50:0.05:0.95', 'maxDets': [1,10,100],
            'qualitative_threshold': 0.25, 'qualitative_iou': 0.5,
            'raw_phrase_token_threshold': 0.1}


def decode_predictions(outputs, batch, geometry, tokenizer):
    import torch
    from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps
    validate_predictions(outputs.logits, outputs.pred_boxes, batch['attention_mask'])
    maps = build_label_maps(outputs.logits, batch['input_ids'])
    predictions, raw = [], []
    for i, (mapping, frame) in enumerate(zip(maps, geometry)):
        scores = outputs.logits[i].sigmoid()
        span_scores = torch.stack([scores[:, m.bool()].mean(-1) for m in mapping], dim=-1)
        values, classes = span_scores.max(-1)
        order = values.argsort(descending=True, stable=True)
        # Transfer once per image; avoid thousands of per-query GPU synchronizations.
        scores=scores.detach().cpu();values=values.detach().cpu();classes=classes.detach().cpu()
        boxes=outputs.pred_boxes[i].detach().cpu()
        ids=batch['input_ids'][i].detach().cpu()
        ordinary=torch.ones_like(ids,dtype=torch.bool)
        for special in tokenizer.all_special_ids + [tokenizer.convert_tokens_to_ids('.')]:
            ordinary &= ids != special
        ow,oh = frame['original_wh']; rw,rh = frame['resized_wh']; pw,ph = frame['padded_wh']
        kept = 0
        for query in order.tolist():
            cx,cy,w,h = boxes[query].tolist()
            x1,x2 = max(0.,min(ow,(cx-w/2)*pw*ow/rw)), max(0.,min(ow,(cx+w/2)*pw*ow/rw))
            y1,y2 = max(0.,min(oh,(cy-h/2)*ph*oh/rh)), max(0.,min(oh,(cy+h/2)*ph*oh/rh))
            valid = x2>x1 and y2>y1
            selected = (scores[query,:len(ids)] > PROTOCOL['raw_phrase_token_threshold']) & ordinary
            # Ordinary text only, retaining incomplete/unmapped raw phrases honestly.
            phrase = tokenizer.decode(ids[selected])
            row = {'image_id': frame['image_id'], 'query_id': query, 'category_id': int(classes[query]),
                   'score': float(values[query]), 'bbox': [x1,y1,x2-x1,y2-y1], 'raw_phrase': phrase,
                   'valid_clipped_box': valid}
            raw.append(row)
            if valid and kept < 100:
                predictions.append({k:row[k] for k in ('image_id','category_id','score','bbox')})
                kept += 1
    return predictions, raw


def coco_metrics(data, predictions):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = copy.deepcopy(data); gt.createIndex()
        if predictions:
            dt = gt.loadRes(copy.deepcopy(predictions))
        else:
            dt = COCO(); dt.dataset = {'images':copy.deepcopy(data['images']),
                                     'categories':copy.deepcopy(data['categories']), 'annotations':[]}; dt.createIndex()
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = [i['id'] for i in data['images']]
        ev.params.catIds = list(range(8))
        ev.evaluate(); ev.accumulate(); ev.summarize()
    def measured(value):
        return float(value) if np.isfinite(value) and value >= 0 else None
    names = ('AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large')
    result = dict(zip(names, map(measured, ev.stats)))
    result['classes'] = {}
    for k, name in enumerate(LABELS):
        precision = ev.eval['precision'][:,:,k,0,2]
        recall = ev.eval['recall'][:,k,0,2]
        objects = [a for a in data['annotations'] if a['category_id']==k]
        result['classes'][name] = {'objects':len(objects), 'images':len({a['image_id'] for a in objects}),
            'AP':measured(precision[precision>=0].mean()) if (precision>=0).any() else None,
            'AR100':measured(recall[recall>=0].mean()) if (recall>=0).any() else None}
    return result


def annotation_errors(data, predictions):
    """Deterministic annotations-relative diagnostics, never human photo judgments."""
    def iou(a,b):
        x,y,w,h=a; u,v,s,t=b
        intersection=max(0,min(x+w,u+s)-max(x,u))*max(0,min(y+h,v+t)-max(y,v))
        return intersection/(w*h+s*t-intersection) if intersection else 0.
    result=[]
    for image in data['images']:
        gt=[a for a in data['annotations'] if a['image_id']==image['id']]
        det=sorted([p for p in predictions if p['image_id']==image['id'] and p['score']>=.25],key=lambda p:-p['score'])
        matched=set(); issues=[]
        for p in det:
            candidates=[(iou(p['bbox'],a['bbox']),j) for j,a in enumerate(gt)
                        if j not in matched and a['category_id']==p['category_id']]
            best=max(candidates, default=(0,-1))
            if best[0]>=.5:
                matched.add(best[1]); continue
            overlaps=[(iou(p['bbox'],a['bbox']),j) for j,a in enumerate(gt)]
            overlap,j=max(overlaps,default=(0,-1))
            kind=('duplicate' if j in matched and gt[j]['category_id']==p['category_id'] else 'wrong_class') if overlap>=.5 else 'unmatched_prediction'
            issues.append({'kind':kind,'prediction':p})
        issues.extend({'kind':'missed_annotation','annotation_id':a['id'],'category_id':a['category_id']}
                      for j,a in enumerate(gt) if j not in matched)
        result.append({'image_id':image['id'],'matched':len(matched),'issues':issues})
    return result


def evaluate(model, processor, adapter, dataset, device, shortest, longest):
    import torch
    from tqdm import tqdm
    previous = model.training
    started=time.perf_counter()
    model.eval()
    predictions=[]; diagnostics=[]; losses=[]
    try:
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc='validation', unit='image'):
                batch, geometry = get_batch(dataset,[i],processor,shortest,longest)
                batch = move_batch(batch,device)
                out = adapter(model,batch)
                checked_loss(out,model.config)
                losses.append(float(out.loss))
                pred, raw = decode_predictions(out.outputs,batch,geometry,processor.tokenizer)
                predictions.extend(pred); diagnostics.extend(raw)
        return {'metrics':coco_metrics(dataset.data,predictions),'mean_loss_per_image':sum(losses)/len(losses),
                'elapsed_seconds_including_loading_loss_decoding_and_metrics':time.perf_counter()-started,
                'predictions':predictions,'raw_queries':diagnostics,
                'annotation_relative_errors':annotation_errors(dataset.data,predictions),
                'human_visual_review':'pending; annotations-relative errors are not verified photo judgments'}
    finally:
        model.train(previous)
