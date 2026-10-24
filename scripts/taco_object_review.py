"""Non-exclusive per-GT candidate diagnostics over validated saved rows."""
from .taco_error_analysis import iou
from .training_smoke import LABELS


def eligible_rows(rows, threshold=.05, source='all 900 (diagnostic)'):
    if not 0 <= threshold <= 1:
        raise ValueError('Diagnostic threshold must be in [0,1]')
    if source not in ('all 900 (diagnostic)', 'official top100'):
        raise ValueError('Unknown diagnostic candidate source')
    return [r for r in rows if r['valid_clipped_box'] and r['score'] >= threshold
            and (source == 'all 900 (diagnostic)' or r['retained'])]


def candidate(row, ann):
    return {'query_id':row['query_id'], 'predicted_class':LABELS[row['category_id']],
            'category_id':row['category_id'], 'score':row['score'],
            'iou':iou(row['bbox'], ann['bbox']), 'bbox':list(row['bbox']),
            'raw_phrase':row['raw_phrase'], 'rank':row['rank'], 'retained':row['retained'],
            'same_class_as_annotation':row['category_id'] == ann['category_id']}


def annotation_review(ann, rows, threshold=.05, source='all 900 (diagnostic)'):
    group = eligible_rows(rows, threshold, source)
    overlaps = sorted((candidate(r,ann) for r in group),
                      key=lambda r:(-r['iou'],-r['score'],r['query_id']))
    # Best overlap means positive intersection; no overlap is explicitly absent.
    overlaps = [r for r in overlaps if r['iou'] > 0]
    same = [r for r in overlaps if r['same_class_as_annotation']]
    best_any = overlaps[0] if overlaps else None
    best_same = same[0] if same else None
    any_coverage = bool(best_any and best_any['iou'] >= .5)
    same_coverage = bool(best_same and best_same['iou'] >= .5)
    all_valid = [r for r in rows if r['valid_clipped_box']]
    same_unfiltered = [r for r in all_valid if r['category_id']==ann['category_id'] and iou(r['bbox'],ann['bbox'])>=.5]
    kind = ('correct_class_coverage' if same_coverage else
            'localized_without_correct_class_at_threshold' if any_coverage else
            'no_qualifying_localization_at_threshold')
    return {'annotation_id':ann['id'], 'image_id':ann['image_id'],
            'ground_truth_class':LABELS[ann['category_id']], 'bbox':list(ann['bbox']),
            'threshold':threshold, 'source':source, 'iou_threshold':.5,
            'stored_queries':len(rows), 'eligible_queries':len(group),
            'best_any':best_any, 'best_same':best_same,
            'any_class_coverage':any_coverage, 'correct_class_coverage':same_coverage,
            'same_class_coverage_all900_no_score_floor':bool(same_unfiltered),
            'qualifying_same_class_candidates':sum(r['iou']>=.5 for r in same),
            'qualifying_any_class_candidates':sum(r['iou']>=.5 for r in overlaps),
            'overlapping_candidates':overlaps, 'kind':kind,
            'human_confirmation':'pending; class agreement is annotation-relative'}


def object_limits(ann, width, height):
    x,y,w,h=ann['bbox']
    margin=max(w,h,16)*1.5
    return (max(0,x-margin),min(width,x+w+margin)), (min(height,y+h+margin),max(0,y-margin))
