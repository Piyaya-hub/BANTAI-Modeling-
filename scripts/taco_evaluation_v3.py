"""Stage 5 nine-class evaluator supporting legacy-47, annex-18, and combined-65 views.

Evaluates on the 9-class waste taxonomy including paper (id: 8).
Protocol: no-NMS COCO bbox evaluation across 0.50:0.05:0.95 IoU thresholds.
"""
from __future__ import annotations

import contextlib
import copy
import io
import time
from typing import Any, Dict, List, Optional

from .taco_training_labels_v2 import LABELS
from .taco_training_data_v2 import get_batch

PROTOCOL = {
    "version": "bantai.taco-evaluation.v3",
    "box_frame": "valid_image_normalized_cxcywh",
    "class_score": "mean_ordinary_subword_sigmoid",
    "query_class": "single_argmax",
    "score_filter": 0.0,
    "max_predictions_per_image": 100,
    "nms": False,
    "iou": "COCO_0.50:0.05:0.95",
    "maxDets": [1, 10, 100],
    "qualitative_threshold": 0.25,
    "qualitative_iou": 0.5,
    "raw_phrase_token_threshold": 0.1,
}


def decode_predictions(outputs, batch, geometry, tokenizer):
    import torch
    from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps
    from .training_smoke import validate_predictions

    validate_predictions(outputs.logits, outputs.pred_boxes, batch["attention_mask"])
    maps = build_label_maps(outputs.logits, batch["input_ids"])
    predictions, raw = [], []
    for i, (mapping, frame) in enumerate(zip(maps, geometry)):
        scores = outputs.logits[i].sigmoid()
        span_scores = torch.stack([scores[:, m.bool()].mean(-1) for m in mapping], dim=-1)
        values, classes = span_scores.max(-1)
        order = values.argsort(descending=True, stable=True)

        scores = scores.detach().cpu()
        values = values.detach().cpu()
        classes = classes.detach().cpu()
        boxes = outputs.pred_boxes[i].detach().cpu()
        ids = batch["input_ids"][i].detach().cpu()

        ordinary = torch.ones_like(ids, dtype=torch.bool)
        for special in tokenizer.all_special_ids + [tokenizer.convert_tokens_to_ids(".")]:
            ordinary &= ids != special

        ow, oh = frame["original_wh"]
        kept = 0
        for query in order.tolist():
            cx, cy, w, h = boxes[query].tolist()
            x1 = max(0.0, min(ow, (cx - w / 2.0) * ow))
            x2 = max(0.0, min(ow, (cx + w / 2.0) * ow))
            y1 = max(0.0, min(oh, (cy - h / 2.0) * oh))
            y2 = max(0.0, min(oh, (cy + h / 2.0) * oh))
            valid = x2 > x1 and y2 > y1
            selected = (scores[query, : len(ids)] > PROTOCOL["raw_phrase_token_threshold"]) & ordinary
            phrase = tokenizer.decode(ids[selected])
            row = {
                "image_id": frame["image_id"],
                "query_id": query,
                "category_id": int(classes[query]),
                "score": float(values[query]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "raw_phrase": phrase,
                "valid_clipped_box": valid,
            }
            raw.append(row)
            if valid and kept < 100:
                predictions.append({k: row[k] for k in ("image_id", "category_id", "score", "bbox")})
                kept += 1
    return predictions, raw


def coco_metrics(data, predictions, img_ids=None, cat_ids=None, label_names=None):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if cat_ids is None:
        cat_ids = list(range(len(LABELS)))
    if label_names is None:
        label_names = [LABELS[i] for i in cat_ids]
    if img_ids is None:
        img_ids = [i["id"] for i in data["images"]]

    filtered_images = [img for img in data["images"] if img["id"] in img_ids]
    filtered_annotations = [ann for ann in data["annotations"] if ann["image_id"] in img_ids and ann["category_id"] in cat_ids]
    sub_data = {
        "info": copy.deepcopy(data.get("info", {})),
        "images": copy.deepcopy(filtered_images),
        "annotations": copy.deepcopy(filtered_annotations),
        "categories": [c for c in data["categories"] if c["id"] in cat_ids],
    }

    sub_predictions = [p for p in predictions if p["image_id"] in img_ids and p["category_id"] in cat_ids]

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = sub_data
        gt.createIndex()
        if sub_predictions:
            dt = gt.loadRes(copy.deepcopy(sub_predictions))
        else:
            dt = COCO()
            dt.dataset = {
                "images": copy.deepcopy(filtered_images),
                "categories": copy.deepcopy(sub_data["categories"]),
                "annotations": [],
            }
            dt.createIndex()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = img_ids
        ev.params.catIds = cat_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    def measured(value):
        return float(value) if np.isfinite(value) and value >= 0 else None

    names = (
        "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
    )
    result = dict(zip(names, map(measured, ev.stats)))
    result["classes"] = {}
    for idx, (cid, name) in enumerate(zip(cat_ids, label_names)):
        precision = ev.eval["precision"][:, :, idx, 0, 2]
        recall = ev.eval["recall"][:, idx, 0, 2]
        objs = [a for a in filtered_annotations if a["category_id"] == cid]
        result["classes"][name] = {
            "objects": len(objs),
            "images": len({a["image_id"] for a in objs}),
            "AP": measured(precision[precision >= 0].mean()) if (precision >= 0).any() else None,
            "AR100": measured(recall[recall >= 0].mean()) if (recall >= 0).any() else None,
        }
    return result


def evaluate_stage5(model, processor, adapter, dataset, device, shortest=256, longest=384):
    """
    Evaluates across all 65 validation images and computes:
      1. legacy_47 (original 47 images, categories 0..7)
      2. annex_18 (18 new paper images, categories 0..8)
      3. combined_65 (all 65 images, categories 0..8)
    """
    import torch
    from tqdm import tqdm
    from .training_smoke import move_batch, checked_loss

    previous = model.training
    started = time.perf_counter()
    model.eval()

    # Load frozen original 47 validation IDs to separate views
    from .configuration import ROOT, strict_load
    old_stage2_path = ROOT / "exports/taco/strict-clean-stage2-v1/validation.json"
    old_stage2_data = strict_load(old_stage2_path)
    legacy_47_ids = set(i["id"] for i in old_stage2_data["images"])
    all_65_ids = set(i["id"] for i in dataset.data["images"])
    annex_18_ids = all_65_ids - legacy_47_ids

    assert len(legacy_47_ids) == 47
    assert len(annex_18_ids) == 18
    assert len(all_65_ids) == 65

    predictions, diagnostics, losses = [], [], []
    try:
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc="validation (65 images)", unit="image"):
                batch, geometry = get_batch(dataset, [i], processor, shortest, longest)
                batch = move_batch(batch, device)
                out = adapter(model, batch)
                checked_loss(out, model.config)
                losses.append(float(out.loss))
                pred, raw = decode_predictions(out.outputs, batch, geometry, processor.tokenizer)
                predictions.extend(pred)
                diagnostics.extend(raw)

        # 1. Combined 65 (9 classes)
        combined_metrics = coco_metrics(
            dataset.data,
            predictions,
            img_ids=sorted(all_65_ids),
            cat_ids=list(range(9)),
        )

        # 2. Legacy 47 (8 classes, for comparison with Stage 4)
        legacy_conditioned_metrics = coco_metrics(
            dataset.data,
            predictions,
            img_ids=sorted(legacy_47_ids),
            cat_ids=list(range(8)),
            label_names=list(LABELS[:8]),
        )

        # 3. Annex 18 (9 classes, paper evaluation)
        annex_metrics = coco_metrics(
            dataset.data,
            predictions,
            img_ids=sorted(annex_18_ids),
            cat_ids=list(range(9)),
        )

        # Nine-class prompt/argmax filtering is not the Stage4 eight-class
        # protocol. Evaluate the frozen47 again with its original prompt/adapter.
        from .taco_evaluation import evaluate as evaluate_legacy
        from .training_adapter import EncoderOutputAdapter
        class LegacyView:
            data = old_stage2_data
            images = old_stage2_data["images"]
            def __len__(self):return len(self.images)
            def __getitem__(self,index):
                ident=self.images[index]["id"]
                image,target=dataset[next(i for i,row in enumerate(dataset.images) if row["id"]==ident)]
                target={"image_id":ident,"annotations":[a for a in self.data["annotations"] if a["image_id"]==ident]}
                return image,target
        legacy_value=evaluate_legacy(model,processor,EncoderOutputAdapter(model,processor.tokenizer),LegacyView(),device,shortest,longest)
        elapsed = time.perf_counter() - started
        return {
            "combined_65": combined_metrics,
            "legacy_47": legacy_value["metrics"],
            "legacy_47_evaluation": legacy_value,
            "legacy_47_nine_class_conditioning": legacy_conditioned_metrics,
            "legacy_47_protocol": "bantai.taco-evaluation.v2; original eight-class prompt; recorded input scale",
            "validation_forwards": len(dataset)+47,
            "protocol": PROTOCOL,
            "annex_18": annex_metrics,
            "mean_loss_per_image": sum(losses) / len(losses),
            "elapsed_seconds": elapsed,
            "predictions": predictions,
            "raw_queries": diagnostics,
        }
    finally:
        model.train(previous)
