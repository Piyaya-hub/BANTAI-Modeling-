"""Opt-in display/audit suppression of saved queries, never official evaluation."""
from .taco_error_analysis import iou

NMS_MODES = ('none (raw queries)', 'class-agnostic IoU 0.50', 'class-aware IoU 0.50',
             'agnostic IoU 0.50 + containment 0.80',
             'class-aware IoU 0.50 + containment 0.80')


def intersection_over_minimum_area(a,b):
    """Symmetric IoMin for original-pixel xywh boxes; zero for empty boxes."""
    x,y,w,h=a;u,v,s,t=b
    if min(w,h,s,t)<=0:return 0.
    intersection=max(0,min(x+w,u+s)-max(x,u))*max(0,min(y+h,v+t)-max(y,v))
    return intersection/min(w*h,s*t)


def nms_with_proof(rows, nms_mode=NMS_MODES[0]):
    """Greedy score descending/query-ID ascending; inclusive IoU .50 boundary.

    Input rows are valid passing queries from one image. Raw mode preserves their
    original order. Rows, scores, boxes, class IDs and original ranks never change.
    A suppressed row cannot suppress any other row (no transitive grouping).
    """
    if nms_mode not in NMS_MODES:raise ValueError('Unknown diagnostic NMS mode')
    if nms_mode==NMS_MODES[0]:return list(rows),[]
    kept=[];suppressed=[]
    for row in sorted(rows,key=lambda r:(-r['score'],r['query_id'])):
        for prior in kept:
            if nms_mode.startswith('class-aware') and prior['category_id']!=row['category_id']:continue
            overlap=iou(row['bbox'],prior['bbox'])
            containment=intersection_over_minimum_area(row['bbox'],prior['bbox']) if 'containment' in nms_mode else None
            if overlap>=.50 or (containment is not None and containment>=.80):
                proof={'query_id':row['query_id'],'suppressed_by_query_id':prior['query_id'],
                       'iou':overlap,'same_predicted_class':prior['category_id']==row['category_id']}
                if containment is not None:proof['iomin']=containment
                suppressed.append(proof)
                break
        else:kept.append(row)
    return kept,suppressed


def diagnostic_nms(rows,nms_mode=NMS_MODES[0]):
    return nms_with_proof(rows,nms_mode)[0]


io_min = intersection_over_minimum_area
