"""D-081 bounded 9-vs-10-prompt showcase on four non-held-out images only."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import uuid

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.taco_dataset import image_path, sha256
from scripts.taco_training_labels_v2 import LABELS, PROMPT
from scripts.training_smoke import state_hash

IMAGE_IDS = (68, 72, 92, 841)
ANNOTATION_CLASS = {"Food waste": 9, "Food Can": 5, "Plastic film": 4,
                    "Disposable food container": 6}
CLASS_NAMES = (*LABELS, "food_waste")
# Keep the separator after ``paper``. Removing PROMPT's final period would merge
# the last two labels and make Transformers expose only nine phrase spans.
PROMPT_10 = PROMPT + " food waste."
EXPECTED_PARAMETER_SHA256 = "cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422"
TIER2_RUN = ROOT / "runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14"
CHECKPOINT = TIER2_RUN / "checkpoints/best-inference"
D080_SUMMARY = ROOT / "runs/training/stage5-organic-audit-230f56adfa31496fb475d439ff588f20/summary.json"
RIGHTS_PATH = ROOT / "work/stage5-paper/rights_organic_showcase.json"


def xywh_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _write_rights_before_decode(source, images, cached, frozen, source_hash, root):
    """Bind cached verified photo hashes before the first photo bytes are read."""
    if set(IMAGE_IDS) & (frozen["validation"] | frozen["test"]):
        raise ValueError("Showcase IDs overlap a frozen validation/test split")
    records = []
    for image_id in IMAGE_IDS:
        image = images[image_id]
        evidence = cached["physical_food_image_checks"].get(str(image_id))
        if (not evidence or not evidence.get("fresh_matches_cache")
                or evidence.get("status") != "readable" or evidence.get("exif_orientation") != 1
                or evidence.get("issues")):
            raise ValueError(f"D-080 physical verification is not clean for image {image_id}")
        records.append({"image_id": image_id, "file_name": image["file_name"],
                        "width": image["width"], "height": image["height"],
                        "sha256": evidence["file_sha256"], "split": "non-held-out / proposed train candidate"})
    record = {
        "schema_version": "bantai.organic-showcase-photo-binding.v1",
        "authorization": "D-081",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "local non-held-out diagnostic display/evaluation only",
        "rights_status": "workflow authorization only; rightsholder permission/institutional clearance/definitive legal clearance not claimed",
        "photo_rights_cleared": False,
        "annotation_database_rights_cleared": False,
        "source_annotations_sha256": source_hash,
        "frozen_split_counts": {name: len(ids) for name, ids in frozen.items()},
        "frozen_validation_ids_sha256": hashlib.sha256(json.dumps(sorted(frozen["validation"])).encode()).hexdigest(),
        "frozen_test_ids_sha256": hashlib.sha256(json.dumps(sorted(frozen["test"])).encode()).hexdigest(),
        "test_json_opened": False,
        "heldout_photos_opened": 0,
        "photos": records,
    }
    RIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RIGHTS_PATH.exists():
        existing = json.loads(RIGHTS_PATH.read_text(encoding="utf-8"))
        comparable = dict(existing)
        comparable.pop("created_utc", None)
        candidate = dict(record)
        candidate.pop("created_utc", None)
        if comparable != candidate:
            raise ValueError("Existing Organic showcase rights manifest differs; refusing overwrite")
        record = existing
    else:
        RIGHTS_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record, hashlib.sha256(RIGHTS_PATH.read_bytes()).hexdigest()


def _map_ground_truth(source, images):
    cats = {row["id"]: row["name"] for row in source["categories"]}
    by_image = defaultdict(list)
    for ann in source["annotations"]:
        if ann["image_id"] not in IMAGE_IDS:
            continue
        name = cats[ann["category_id"]]
        if name not in ANNOTATION_CLASS:
            continue
        specific = ANNOTATION_CLASS[name]
        general_id = {9: 2, 5: 3, 4: 0, 6: 0}[specific]
        by_image[ann["image_id"]].append({
            "id": ann["id"], "image_id": ann["image_id"], "category_id": specific,
            "specific_category_id": specific, "specific_class": CLASS_NAMES[specific],
            "mapped_category_id": general_id,
            "mapped_category": {0: "Plastic", 2: "Organic", 3: "Metal"}[general_id],
            "material": {9: "organic_biodegradable", 5: "metal", 4: "plastic", 6: "plastic"}[specific],
            "hazard_flag": False, "source_category": name,
            "bbox": ann["bbox"], "area": ann.get("area"), "iscrowd": ann.get("iscrowd", 0),
        })
    expected = {68: {163, 162}, 72: {171, 170}, 92: {244}, 841: {2611, 2610}}
    for image_id, ids in expected.items():
        if {row["id"] for row in by_image[image_id]} != ids:
            raise ValueError(f"Unexpected ground-truth annotations for Organic showcase image {image_id}")
    return by_image


def _prediction_records(output, input_batch, tokenizer, frame, expected_classes):
    import torch
    from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps

    logits = output.logits
    boxes = output.pred_boxes
    if logits.ndim != 3 or logits.shape[:2] != (1, 900) or boxes.shape != (1, 900, 4):
        raise ValueError(f"Unexpected detector output shape: logits={tuple(logits.shape)} boxes={tuple(boxes.shape)}")
    maps = build_label_maps(logits, input_batch["input_ids"])[0]
    if len(maps) != expected_classes:
        raise ValueError(f"Prompt has {len(maps)} category spans; expected {expected_classes}")
    token_ids = input_batch["input_ids"][0].detach().cpu()
    span_names = [tokenizer.decode(token_ids[mask[:len(token_ids)].bool().detach().cpu()]).strip().rstrip(".")
                  for mask in maps]
    expected_names = [name.replace("_", " ") for name in CLASS_NAMES[:expected_classes]]
    if [name.lower() for name in span_names] != expected_names:
        raise ValueError(f"Prompt/category spans mismatch: {span_names}")

    probs = logits[0].sigmoid()
    class_scores = torch.stack([probs[:, mask[:probs.shape[1]].bool()].mean(-1) for mask in maps], dim=-1)
    top_scores, top_classes = class_scores.max(-1)
    order = torch.argsort(top_scores, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(1, len(order) + 1, device=order.device)
    ow, oh = frame["original_wh"]
    boxes_cpu = boxes[0].detach().cpu().tolist()
    score_rows = class_scores.detach().cpu().tolist()
    top_scores_cpu = top_scores.detach().cpu().tolist()
    top_classes_cpu = top_classes.detach().cpu().tolist()
    ranks_cpu = rank.detach().cpu().tolist()
    records = []
    bantai_names = {0: "Potentially Hazardous", 1: "Plastic", 2: "Plastic",
                    3: "Plastic", 4: "Plastic", 5: "Metal", 6: "Plastic",
                    7: "Plastic", 8: "Paper", 9: "Organic"}
    bantai_ids = {0: 4, 1: 0, 2: 0, 3: 0, 4: 0, 5: 3, 6: 0, 7: 0, 8: 1, 9: 2}
    for q in range(900):
        cx, cy, bw, bh = boxes_cpu[q]
        x1, y1 = max(0.0, (cx - bw / 2) * ow), max(0.0, (cy - bh / 2) * oh)
        x2, y2 = min(float(ow), (cx + bw / 2) * ow), min(float(oh), (cy + bh / 2) * oh)
        valid = x2 > x1 and y2 > y1
        cls = int(top_classes_cpu[q])
        records.append({"query_id": q, "category_id": cls, "specific_category_id": cls,
                        "specific_class": CLASS_NAMES[cls], "score": float(top_scores_cpu[q]),
                        "bantai_category_id": bantai_ids[cls], "bantai_category": bantai_names[cls],
                        "rank": int(ranks_cpu[q]), "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "valid_clipped_box": valid,
                        "class_score_vector": [float(x) for x in score_rows[q]]})
    return records, span_names


def _gt_diagnostics(gt_rows, predictions_9, predictions_10):
    result = []
    for gt in gt_rows:
        box = gt["bbox"]
        entry = {"annotation_id": gt["id"], "image_id": gt["image_id"],
                 "source_category": gt["source_category"],
                 "specific_category_id": gt["specific_category_id"],
                 "bantai_category": gt["mapped_category"], "bbox_xywh": box}
        for prompt_name, rows in (("9_prompt", predictions_9), ("10_prompt", predictions_10)):
            overlaps = [(xywh_iou(row["bbox"], box), row) for row in rows if row["valid_clipped_box"]]
            best_iou, best_row = max(overlaps, key=lambda pair: (pair[0], -pair[1]["query_id"]))
            entry[prompt_name] = {"best_any_iou_query": {
                "query_id": best_row["query_id"], "predicted_class": best_row["specific_class"],
                "score": best_row["score"], "iou": best_iou, "rank_by_top1_score": best_row["rank"]}}
            same = [(iou, row) for iou, row in overlaps if row["category_id"] == gt["specific_category_id"]]
            if same:
                same_iou, same_row = max(same, key=lambda pair: (pair[0], -pair[1]["query_id"]))
                entry[prompt_name]["best_same_class_iou_query"] = {
                    "query_id": same_row["query_id"], "score": same_row["score"],
                    "iou": same_iou, "rank_by_top1_score": same_row["rank"]}
        organic_rows = predictions_10
        organic_sorted = sorted(organic_rows, key=lambda row: (-row["class_score_vector"][9], row["query_id"]))
        organic_rank = {row["query_id"]: i + 1 for i, row in enumerate(organic_sorted)}
        best_organic = max(organic_rows, key=lambda row: (row["class_score_vector"][9], -row["query_id"]))
        entry["10_prompt_best_organic_query"] = {
            "query_id": best_organic["query_id"], "score": best_organic["class_score_vector"][9],
            "iou": xywh_iou(best_organic["bbox"], box), "rank_by_organic_score": organic_rank[best_organic["query_id"]],
            "top1_predicted_class": best_organic["specific_class"], "top1_score": best_organic["score"]}
        localized = [(xywh_iou(row["bbox"], box), row) for row in organic_rows
                     if row["valid_clipped_box"] and xywh_iou(row["bbox"], box) >= 0.50]
        best_organic_iou, best_organic_localized = max(
            localized, key=lambda pair: (pair[1]["class_score_vector"][9], pair[0], -pair[1]["query_id"]))
        entry["10_prompt_best_organic_localized_query"] = {
            "query_id": best_organic_localized["query_id"],
            "score": best_organic_localized["class_score_vector"][9],
            "iou": best_organic_iou,
            "rank_by_organic_score": organic_rank[best_organic_localized["query_id"]],
            "top1_predicted_class": best_organic_localized["specific_class"],
            "top1_score": best_organic_localized["score"]}
        entry["10_prompt_queries_iou_ge_0_50"] = [
            {"query_id": row["query_id"], "iou": iou, "predicted_class": row["specific_class"],
             "top1_score": row["score"], "class_scores": dict(zip(CLASS_NAMES, row["class_score_vector"]))}
            for iou, row in sorted(((xywh_iou(row["bbox"], box), row) for row in predictions_10
                                    if row["valid_clipped_box"]), key=lambda pair: (-pair[0], pair[1]["query_id"]))
            if iou >= 0.50]
        result.append(entry)
    return result


def main():
    import torch
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    from scripts.configuration import strict_load
    from scripts.training_smoke import loading_ok as verify_loading

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not torch.cuda.is_available():
        raise RuntimeError("D-081 requires CUDA; refusing CPU fallback or extra inference")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce GTX 1660 Ti":
        raise RuntimeError("Unexpected CUDA device; stop before photo decoding")

    manifest = strict_load(CHECKPOINT / "soup_manifest.json")
    if (manifest.get("parameter_sha256") != EXPECTED_PARAMETER_SHA256
            or manifest.get("authorization") != "D-076"
            or manifest.get("resume_supported") is not False):
        raise ValueError("D-076 champion manifest verification failed")
    for name, expected in manifest["files"].items():
        if sha256(CHECKPOINT / name) != expected:
            raise ValueError(f"D-076 checkpoint file hash mismatch: {name}")

    stage = ROOT / "exports/taco/strict-clean-stage5-v1"
    prep_manifest = strict_load(stage / "preparation_manifest.json")
    source_path = Path(prep_manifest["source"]["annotations"])
    dataset_root = Path(prep_manifest["source"]["dataset_root"]).resolve()
    source_bytes_hash = sha256(source_path)
    source = strict_load(source_path)
    cached = strict_load(D080_SUMMARY)
    images = {row["id"]: row for row in source["images"]}
    ledger = [json.loads(line) for line in (stage / "image_eligibility.jsonl").read_text(encoding="utf-8").splitlines()]
    frozen = {split: {row["source_image_id"] for row in ledger
                      if row.get("eligible") and row.get("split") == split}
              for split in ("validation", "test")}
    if len(frozen["validation"]) != 65 or len(frozen["test"]) != 65:
        raise ValueError("Frozen split membership ledger is incomplete")
    if set(IMAGE_IDS) & (frozen["validation"] | frozen["test"]):
        raise ValueError("Showcase IDs overlap frozen validation/test")
    if any(image_id not in images for image_id in IMAGE_IDS):
        raise ValueError("Showcase source image missing from raw metadata")
    for image_id in IMAGE_IDS:
        expected_file = cached["physical_food_image_checks"][str(image_id)]["file_sha256"]
        if images[image_id]["file_name"] != next(
                p["file_name"] for p in cached["ledger"] if p["image_id"] == image_id and p["tier"] == "tier2"):
            raise ValueError(f"D-080 file name differs for image {image_id}")
        if image_id not in cached["tiers"]["tier2"]["image_ids"]:
            raise ValueError(f"Image {image_id} is not a clean Tier 2 candidate")
    gt = _map_ground_truth(source, images)
    out = ROOT / "runs/training" / ("stage5-organic-hybrid-" + uuid.uuid4().hex)
    out.mkdir(parents=True)

    # This manifest is durable before any source image byte read or decode.
    rights, rights_sha = _write_rights_before_decode(source, images, cached, frozen,
                                                       source_bytes_hash, ROOT)
    bound = {row["image_id"]: row["sha256"] for row in rights["photos"]}
    photo_paths = {image_id: image_path(dataset_root, images[image_id]["file_name"])
                   for image_id in IMAGE_IDS}
    heldout_paths = set()
    for row in ledger:
        if row.get("eligible") and row.get("split") in ("validation", "test"):
            image = images[row["source_image_id"]]
            heldout_paths.add(image_path(dataset_root, image["file_name"]).resolve())

    def access_guard(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name.lower() == "test.json" or path in heldout_paths:
            raise AssertionError("D-081 blocked test JSON or frozen validation/test photo access")
        if path.is_relative_to(dataset_root) and path not in set(photo_paths.values()):
            raise AssertionError("D-081 photo access outside the four-image showcase")
    sys.addaudithook(access_guard)

    # Verify photo bytes against the already-written binding, then decode exactly these four.
    decoded_images = {}
    photo_reads = []
    for image_id in IMAGE_IDS:
        path = photo_paths[image_id]
        file_hash = sha256(path)
        if file_hash != bound[image_id]:
            raise ValueError(f"Showcase photo SHA-256 differs from pre-decode binding for {image_id}")
        with Image.open(path) as photo:
            photo.load()
            orientation = photo.getexif().get(274, 1)
            if photo.size != (images[image_id]["width"], images[image_id]["height"]) or orientation != 1:
                raise ValueError(f"Showcase dimensions/EXIF changed for image {image_id}")
            decoded_images[image_id] = photo.convert("RGB")
        photo_reads.append({"image_id": image_id, "file_name": images[image_id]["file_name"],
                            "sha256": file_hash, "dimensions": [images[image_id]["width"], images[image_id]["height"]],
                            "exif_orientation": orientation})

    model, loading = GroundingDinoForObjectDetection.from_pretrained(
        CHECKPOINT, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.float32, output_loading_info=True)
    verify_loading(loading)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, local_files_only=True, trust_remote_code=False)
    if state_hash(model) != EXPECTED_PARAMETER_SHA256:
        raise ValueError("Loaded D-076 champion parameter hash mismatch")
    model.to(device="cuda", dtype=torch.float32)
    model.eval()
    forward_count = {"total": 0}

    def count_forward(_module, _inputs):
        forward_count["total"] += 1
        if forward_count["total"] > 8:
            raise AssertionError("D-081 hard limit exceeded: more than eight detector forwards")
    hook = model.register_forward_pre_hook(count_forward)

    results = {}
    gui_predictions = {"Organic Showcase 9-Prompt (384/576)": {},
                       "Organic Showcase 10-Prompt (Hybrid)": {}}
    prompts = (("9_prompt", PROMPT, 9), ("10_prompt", PROMPT_10, 10))
    try:
        for image_id in IMAGE_IDS:
            frame = {"image_id": image_id,
                     "original_wh": [images[image_id]["width"], images[image_id]["height"]]}
            image_result = {"image_id": image_id, "file_name": images[image_id]["file_name"],
                            "photo_sha256": bound[image_id],
                            "ground_truth": gt[image_id], "passes": {}}
            for pass_name, prompt, classes in prompts:
                inputs = processor(images=decoded_images[image_id], text=prompt, padding=True,
                                   return_tensors="pt", size={"shortest_edge": 384, "longest_edge": 576},
                                   pad_size={"height": 576, "width": 576})
                if (inputs["pixel_values"].dtype != torch.float32
                        or tuple(inputs["pixel_values"].shape) != (1, 3, 576, 576)):
                    raise ValueError("384/576 input must be FP32 and square padded to 576")
                cuda_inputs = {key: value.to("cuda") if torch.is_tensor(value) else value
                               for key, value in inputs.items()}
                with torch.inference_mode():
                    outputs = model(**cuda_inputs)
                expected_count = 2 * IMAGE_IDS.index(image_id) + (1 if pass_name == "9_prompt" else 2)
                if forward_count["total"] != expected_count:
                    raise AssertionError("Unexpected forward count/order")
                rows, span_names = _prediction_records(outputs, cuda_inputs, processor.tokenizer, frame, classes)
                image_result["passes"][pass_name] = {
                    "prompt": prompt, "class_names": list(CLASS_NAMES[:classes]),
                    "span_names": span_names, "queries": rows,
                }
                state = ("Organic Showcase 9-Prompt (384/576)" if pass_name == "9_prompt"
                         else "Organic Showcase 10-Prompt (Hybrid)")
                gui_predictions[state][str(image_id)] = rows
                del outputs, cuda_inputs, inputs
            image_result["gt_diagnostics"] = _gt_diagnostics(
                gt[image_id], image_result["passes"]["9_prompt"]["queries"],
                image_result["passes"]["10_prompt"]["queries"])
            results[image_id] = image_result
        torch.cuda.synchronize()
    finally:
        hook.remove()

    if forward_count["total"] != 8:
        raise AssertionError(f"Expected exactly 8 completed forward calls, observed {forward_count['total']}")
    if not torch.cuda.is_available() or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise AssertionError("Unexpected CUDA/TF32 state after evaluation")

    full = {"authorization": "D-081", "status": "complete", "device": torch.cuda.get_device_name(0),
            "precision": "FP32", "TF32": False, "shortest_edge": 384, "longest_edge": 576,
            "square_pad": 576, "training_updates": 0, "backward_passes": 0,
            "detector_forwards": forward_count["total"], "forwards_per_prompt": {"9_prompt": 4, "10_prompt": 4},
            "test_json_opened": False, "heldout_photos_opened": 0,
            "checkpoint": str(CHECKPOINT), "checkpoint_parameter_sha256": EXPECTED_PARAMETER_SHA256,
            "checkpoint_manifest_sha256": sha256(CHECKPOINT / "soup_manifest.json"),
            "rights_manifest": str(RIGHTS_PATH), "rights_manifest_sha256": rights_sha,
            "photo_reads_and_hashes": photo_reads, "prompt_9": PROMPT,
            "prompt_10": PROMPT_10, "input_source_annotations_sha256": source_bytes_hash,
            "images": {str(k): v for k, v in results.items()}}
    gui = {"authorization": "D-081", "image_ids": list(IMAGE_IDS),
           "dataset_root": str(dataset_root),
           "images": {str(i): {"id": i, "file_name": images[i]["file_name"],
                                "width": images[i]["width"], "height": images[i]["height"],
                                "file_sha256": bound[i]} for i in IMAGE_IDS},
           "ground_truth": {str(i): gt[i] for i in IMAGE_IDS},
           "predictions": gui_predictions}
    full_path, gui_path = out / "evaluation.json", out / "gui-showcase.json"
    full_path.write_text(json.dumps(full, indent=2) + "\n", encoding="utf-8")
    gui_path.write_text(json.dumps(gui, separators=(",", ":")) + "\n", encoding="utf-8")
    result_manifest = {"schema_version": "bantai.stage5-organic-showcase.v1",
                       "authorization": "D-081", "status": "complete",
                       "checkpoint_parameter_sha256": EXPECTED_PARAMETER_SHA256,
                       "rights_manifest_sha256": rights_sha,
                       "evaluation_sha256": sha256(full_path),
                       "gui_showcase_sha256": sha256(gui_path),
                       "image_ids": list(IMAGE_IDS), "detector_forwards": 8,
                       "training_updates": 0, "backward_passes": 0,
                       "test_json_opened": False, "heldout_photos_opened": 0}
    (out / "manifest.json").write_text(json.dumps(result_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "rights_manifest": str(RIGHTS_PATH),
                      "forwards": forward_count["total"], "parameter_sha256": EXPECTED_PARAMETER_SHA256,
                      "gt_diagnostics": {str(i): results[i]["gt_diagnostics"] for i in IMAGE_IDS}}, indent=2))


if __name__ == "__main__":
    main()
