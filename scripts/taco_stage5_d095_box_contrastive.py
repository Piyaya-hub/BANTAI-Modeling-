"""Stage 5 Box-Contrastive GT-Region Margin Fine-Tuning Run (D-095).

Authorizes:
1. Warm-start from verified D-093 Champion (`soup_d093best_d091_5050`, SHA-256 51ce99e6b7f2e9b08a070a7c237aba752b08467a84ffe482367e795dd1ad8fb7).
2. Two-epoch bounded fine-tuning (1,152 updates) on 576-image Tier 3 Core export with omitted-accessory masking.
3. GT-Region Box-Contrastive Margin Loss (positive margin boost + interior wrong-class sub-box suppression).
4. Auxiliary Absent-Class False-Positive Suppression (threshold 0.05 on top-15 queries) and proximal L2 anchor (0.004).
5. 5-state validation evaluation on combined-65 and legacy-47 (560 forwards) with scene-level bidirectional consensus audit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from scripts.configuration import strict_load
from scripts.persistence import atomic_json
from scripts.taco_dataset import sha256
from scripts.taco_training_labels_v2 import LABELS, PROMPT
from scripts.taco_training_data_v2 import CocoTrainingDatasetV2, get_batch
from scripts.taco_evaluation_v3 import decode_predictions, coco_metrics
from scripts.taco_evaluation import evaluate as evaluate_legacy
from scripts.training_adapter import EncoderOutputAdapter
from scripts.training_adapter_v2 import EncoderOutputAdapterV2
from scripts.training_checkpoint import save_checkpoint, verify_checkpoint, epoch_batches, rng_state, restore_rng
from scripts.training_smoke import (SNAPSHOT, checked_loss, loading_ok, move_batch, require_training_environment,
                                    state_hash)
from scripts.train_taco import SOURCES
from scripts.train_taco_stage5 import freeze_backbones, load_verified_stage5_checkpoint
from scripts.taco_weight_integration import average_trainable
from scripts.taco_stage5_material_hierarchy import evaluate_case
from scripts.taco_stage5_consensus_audit import official_ap, strict_counts
from scripts.taco_stage5_consensus_tiebreak import apply_consensus_tiebreak, apply_scene_consensus

TARGET_RUN = ROOT / "runs/training/stage5-d095-box-contrastive-20261102-a"
D095_EXPORT = ROOT / "exports/taco/strict-clean-stage5-tier3-core-v1"
D095_RIGHTS = ROOT / "work/stage5-paper/rights_tier3_core.json"

WARMSTART_BUNDLE = ROOT / "runs/training/stage5-d093-multiepoch-hn-20261102-a/checkpoints/best-inference"
EXPECTED_WARMSTART = "51ce99e6b7f2e9b08a070a7c237aba752b08467a84ffe482367e795dd1ad8fb7"

D093_E3_CHECKPOINT = ROOT / "runs/training/stage5-d093-multiepoch-hn-20261102-a/checkpoints/epoch-03"
EXPECTED_D093_E3 = "9501a1418eb01c7519131e29afc22f09bae9063ec2728bc5762eeefce67aa3ce"

D091_CHAMP_BUNDLE = ROOT / "runs/training/stage5-d091-balanced-tier3-20261101-a/checkpoints/best-inference"
EXPECTED_D091_CHAMP = "4146517bd69837e2e11932e4c3ecee5951cd3e862f414e6f8ba45e1ed20212ee"

FROZEN_BACKBONE_HASH = "161c9814bffe1657ae0debd7e21b7ed1c41f29f3ba43fffe879ab31ba36a0b67"
LEGACY_JSON = ROOT / "exports/taco/strict-clean-stage2-v1/validation.json"

CLASS_COUNTS = {0: 252, 1: 151, 2: 39, 3: 97, 4: 168, 5: 157, 6: 75, 7: 63, 8: 238}
N_REF = 151.0
CLASS_WEIGHTS = {c: min(1.80, max(0.75, math.sqrt(N_REF / CLASS_COUNTS[c]))) for c in range(9)}

PROTECTED_CORE_NAMES = ("train_taco.py", "taco_training_data.py", "taco_evaluation.py",
                        "training_checkpoint.py", "training_adapter.py", "training_smoke.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_snapshot(exclude_run: Path | None = None) -> dict[str, str]:
    exclude_run = Path(exclude_run).resolve() if exclude_run else None
    rows: dict[str, str] = {}
    protected = [ROOT / "config/taco_taxonomy.v1.json", ROOT / "config/taco_taxonomy.v2.json"]
    protected += [ROOT / "scripts" / name for name in PROTECTED_CORE_NAMES]
    for path in protected:
        rows[str(path.relative_to(ROOT))] = _sha(path)
    export_root = ROOT / "exports"
    for path in sorted(export_root.rglob("*")):
        if path.is_file() and path.name.lower() != "test.json":
            rows[str(path.relative_to(ROOT))] = _sha(path)
    rights_root = ROOT / "work/stage5-paper"
    if rights_root.exists():
        for path in sorted(rights_root.rglob("*")):
            if path.is_file():
                rows[str(path.relative_to(ROOT))] = _sha(path)
    run_root = ROOT / "runs/training"
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or "checkpoints" not in path.parts:
            continue
        if exclude_run is not None and path.resolve().is_relative_to(exclude_run):
            continue
        rows[str(path.relative_to(ROOT))] = _sha(path)
    return rows


def verify_snapshot(before: dict[str, str], exclude_run: Path | None) -> dict[str, object]:
    after = protected_snapshot(exclude_run)
    if before != after:
        changed = sorted(set(before) ^ set(after) | {key for key in before.keys() & after.keys()
                                                      if before[key] != after[key]})
        raise AssertionError("Protected artifact hash drift: " + ", ".join(changed[:12]))
    return {
        "files": len(before),
        "unchanged": True,
        "snapshot_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
    }


def _bundle_manifest_hashes(bundle: Path) -> dict[str, object]:
    bundle = Path(bundle)
    manifest_name = "soup_manifest.json" if (bundle / "soup_manifest.json").is_file() else "checkpoint_manifest.json"
    manifest = strict_load(bundle / manifest_name)
    for name, expected in manifest["files"].items():
        if _sha(bundle / name) != expected:
            raise ValueError(f"Checkpoint file changed: {bundle / name}")
    return {
        "manifest": str(bundle / manifest_name),
        "manifest_sha256": _sha(bundle / manifest_name),
        "parameter_sha256": manifest["parameter_sha256"],
    }


def mask_omitted_accessories(image: Image.Image, omitted_bboxes: list, retained_bboxes: list) -> Image.Image:
    """
    In-memory replacement of omitted accessory regions with image mean RGB.
    Pixels overlapping ANY retained 0..8 ground-truth bbox are NEVER masked.
    Never modifies disk.
    """
    if not omitted_bboxes:
        return image
    arr = np.array(image, copy=True)
    H, W = arr.shape[:2]
    mean_rgb = np.mean(arr, axis=(0, 1)).astype(arr.dtype)
    mask = np.zeros((H, W), dtype=bool)

    for om_box in omitted_bboxes:
        ox, oy, ow, oh = om_box
        ox1 = max(0, int(round(ox)))
        oy1 = max(0, int(round(oy)))
        ox2 = min(W, int(round(ox + ow)))
        oy2 = min(H, int(round(oy + oh)))
        if ox2 > ox1 and oy2 > oy1:
            mask[oy1:oy2, ox1:ox2] = True

    for gt_box in retained_bboxes:
        gx, gy, gw, gh = gt_box
        gx1 = max(0, int(round(gx)))
        gy1 = max(0, int(round(gy)))
        gx2 = min(W, int(round(gx + gw)))
        gy2 = min(H, int(round(gy + gh)))
        if gx2 > gx1 and gy2 > gy1:
            mask[gy1:gy2, gx1:gx2] = False

    if not np.any(mask):
        return image
    arr[mask] = mean_rgb
    return Image.fromarray(arr)


class CocoTier3CoreDataset(CocoTrainingDatasetV2):
    """Stage 5 Tier 3 Core dataset loader with optional training omitted-accessory masking."""

    def __init__(self, folder: Path, split: str, rights_path: Path, mask_omitted: bool = False):
        super().__init__(folder, split, rights_path)
        self.mask_omitted = mask_omitted
        self.masked_photo_count = 0

    def __getitem__(self, index: int):
        rgb, target = super().__getitem__(index)
        if self.mask_omitted:
            row = self.images[index]
            omitted = row.get("omitted_accessory_bboxes")
            if omitted:
                gt_boxes = [ann["bbox"] for ann in target["annotations"]]
                masked_rgb = mask_omitted_accessories(rgb, omitted, gt_boxes)
                if masked_rgb is not rgb:
                    self.masked_photo_count += 1
                rgb = masked_rgb
        return rgb, target


def _install_access_guard(dataset_root: Path, allowed: set[Path], run_dir: Path):
    dataset_root = dataset_root.resolve()
    allowed = {p.resolve() for p in allowed}
    run_dir = run_dir.resolve()

    def guard(event, args):
        if event in ("os.remove", "os.unlink", "os.rename", "os.replace", "os.mkdir", "os.rmdir"):
            for value in args:
                if isinstance(value, (str, bytes, os.PathLike)):
                    path = Path(os.fsdecode(value)).resolve()
                    if path.is_relative_to(ROOT) and not path.is_relative_to(run_dir):
                        raise AssertionError("D-095 may create/mutate only its new private run directory")
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name.lower() == "test.json":
            raise AssertionError("D-095 blocks test.json reads and writes")
        if path.is_relative_to(dataset_root) and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            if path not in allowed:
                raise AssertionError("D-095 attempted photo access outside authorized train/validation IDs")
        flags = args[2] if len(args) > 2 else 0
        writing = isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        if writing and path.is_relative_to(ROOT) and not path.is_relative_to(run_dir):
            raise AssertionError("D-095 process write outside its new private run directory")

    sys.addaudithook(guard)


def _runtime_model(device="cuda"):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from scripts.training_smoke import MODEL_ID
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
    adapter = EncoderOutputAdapterV2(model, processor.tokenizer)
    freeze_backbones(model)
    return model, processor, adapter


def _class_score_vectors(outputs, input_ids):
    import torch
    from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps
    maps = build_label_maps(outputs.logits, input_ids)
    if any(len(mapping) != 9 for mapping in maps):
        raise ValueError("Nine prompt spans required for D-095 vectors")
    scores = outputs.logits.sigmoid()
    return torch.stack([
        torch.stack([scores[i, :, mask[:scores.shape[-1]].bool()].mean(-1) for mask in maps[i]], dim=-1)
        for i in range(scores.shape[0])
    ], dim=0)


def evaluate_d095(model, processor, adapter, dataset, device, shortest=384, longest=576,
                  *, forward_counter=None):
    """Run exactly 65 nine-class + 47 separate original-eight prompt forwards."""
    import torch
    from tqdm import tqdm
    start = time.perf_counter()
    previous = model.training
    model.eval()
    predictions, raw_queries, losses, vectors = [], [], [], {}
    try:
        with torch.inference_mode():
            for index in tqdm(range(len(dataset)), desc=f"D-095 validation {shortest}/{longest}", unit="image"):
                batch, geometry = get_batch(dataset, [index], processor, shortest, longest)
                if tuple(batch["pixel_values"].shape) != (1, 3, longest, longest) or batch["pixel_values"].dtype != torch.float32:
                    raise ValueError("D-095 input geometry/precision contract changed")
                batch = move_batch(batch, device)
                result = adapter(model, batch)
                checked_loss(result, model.config)
                losses.append(float(result.loss.detach()))
                pred, raw = decode_predictions(result.outputs, batch, geometry, processor.tokenizer)
                class_vectors = _class_score_vectors(result.outputs, batch["input_ids"])[0].detach().cpu().numpy()
                image_id = geometry[0]["image_id"]
                if class_vectors.shape != (900, 9) or class_vectors.dtype != np.float32:
                    class_vectors = class_vectors.astype(np.float32, copy=False)
                vectors[image_id] = class_vectors
                predictions.extend(pred)
                raw_queries.extend(raw)
                del result, batch
        combined = coco_metrics(dataset.data, predictions, img_ids=sorted(i["id"] for i in dataset.images),
                                cat_ids=list(range(9)), label_names=list(LABELS))

        old_data = strict_load(LEGACY_JSON)
        legacy_ids = {row["id"] for row in old_data["images"]}
        if len(legacy_ids) != 47 or len(old_data["annotations"]) != 75:
            raise ValueError("Frozen legacy validation membership changed")
        if not legacy_ids.issubset({row["id"] for row in dataset.images}):
            raise ValueError("D-095 validation does not contain exact legacy47 IDs")
        by_id = {row["id"]: i for i, row in enumerate(dataset.images)}

        class LegacyView:
            data = old_data
            images = old_data["images"]

            def __len__(self):
                return len(self.images)

            def __getitem__(self, index):
                ident = self.images[index]["id"]
                image, _ = dataset[by_id[ident]]
                return image, {
                    "image_id": ident,
                    "annotations": [a for a in self.data["annotations"] if a["image_id"] == ident]
                }

        legacy = evaluate_legacy(model, processor, EncoderOutputAdapter(model, processor.tokenizer),
                                 LegacyView(), device, shortest, longest)
        if forward_counter is not None and forward_counter() != 112:
            raise AssertionError(f"D-095 validation forward count is {forward_counter()}, expected 112")
        return {
            "metrics": combined,
            "legacy_metrics": legacy["metrics"],
            "mean_loss": sum(losses) / len(losses),
            "predictions": predictions,
            "raw_queries": raw_queries,
            "vectors": vectors,
            "elapsed_seconds": time.perf_counter() - start,
        }
    finally:
        model.train(previous)


def _save_eval(run_dir: Path, name: str, result: dict) -> tuple[Path, dict]:
    vector_path = run_dir / f"scores-{name}.npz"
    sorted_ids = sorted(result["vectors"].keys())
    matrix = np.stack([result["vectors"][i] for i in sorted_ids], axis=0)
    np.savez_compressed(vector_path, image_ids=np.array(sorted_ids, dtype=np.int64), scores=matrix)
    val_data = strict_load(D095_EXPORT / "validation.json")
    m_raw = evaluate_case(val_data, result["raw_queries"], "9-specific", "raw")
    m_mat = evaluate_case(val_data, result["raw_queries"], "5-material", "raw")
    m_5cat = evaluate_case(val_data, result["raw_queries"], "bantai-5-category", "raw")
    m_nms = evaluate_case(val_data, result["raw_queries"], "9-specific", "class_aware_iou_0.50")
    m_5cat_nms = evaluate_case(val_data, result["raw_queries"], "bantai-5-category", "class_aware_iou_0.50")

    saved = {
        "state": name,
        "parameter_sha256": result.get("parameter_sha256"),
        "vector_artifact": vector_path.name,
        "vector_shape": list(matrix.shape),
        "combined_65": {
            "AP": result["metrics"]["AP"],
            "AP50": result["metrics"]["AP50"],
            "AP75": result["metrics"]["AP75"],
            "AP_small": result["metrics"].get("AP_small"),
            "AP_medium": result["metrics"].get("AP_medium"),
            "AP_large": result["metrics"].get("AP_large"),
            "AR100": result["metrics"]["AR100"],
            "per_class": {c: result["metrics"]["classes"][c]["AP"] for c in LABELS},
            "per_class_ar100": {c: result["metrics"]["classes"][c]["AR100"] for c in LABELS},
        },
        "material_hierarchy": {
            "raw_9_specific": m_raw["metrics"],
            "raw_5_material": m_mat["metrics"],
            "raw_bantai_5_category": m_5cat["metrics"],
            "nms_050_9_specific": m_nms["metrics"],
            "nms_050_bantai_5_category": m_5cat_nms["metrics"],
        },
        "legacy_47": {
            "AP": result["legacy_metrics"]["AP"],
            "AP50": result["legacy_metrics"]["AP50"],
            "AP75": result["legacy_metrics"].get("AP75"),
            "AR100": result["legacy_metrics"]["AR100"],
        },
        "raw_queries": result["raw_queries"],
    }
    atomic_json(run_dir / f"{name}.json", saved, replace=True)
    return vector_path, saved


def _run_scene_consensus_audit(val_data: dict, raw_rows: list[dict], scores: dict[int, np.ndarray]) -> dict[str, object]:
    baseline = defaultdict(list)
    for row in raw_rows:
        baseline[row["image_id"]].append(dict(row))
    adjusted = defaultdict(list)
    img_dims = {row["id"]: (row["width"], row["height"]) for row in val_data["images"]}
    total_changed = 0
    total_suppressed = 0

    for image_id in sorted(baseline):
        w_img, h_img = img_dims[image_id]
        score_vecs = scores[image_id]
        adj_rows, changed, supp = apply_scene_consensus(
            baseline[image_id], score_vecs, image_width=w_img, image_height=h_img
        )
        adjusted[image_id] = adj_rows
        total_changed += len(changed)
        total_suppressed += len(supp)

    schemes = ("9-specific", "bantai-5-category")
    thresholds = (.05, .10, .15)
    return {
        "profile": "scene-level bidirectional consensus + interior subbox suppression",
        "changed_queries": total_changed,
        "suppressed_queries": total_suppressed,
        "baseline": {scheme: {"metrics": official_ap(val_data, baseline, scheme),
                              "strict": {str(t): strict_counts(val_data, baseline, scheme, t)
                                         for t in thresholds}}
                     for scheme in schemes},
        "consensus": {scheme: {"metrics": official_ap(val_data, adjusted, scheme),
                               "strict": {str(t): strict_counts(val_data, adjusted, scheme, t)
                                          for t in thresholds}}
                      for scheme in schemes},
    }


def _save_inference_bundle(model, processor, adapter, run_dir: Path, state: str, parameter_sha256: str,
                           eval_path: Path, recipe: dict) -> tuple[Path, dict]:
    from scripts.training_smoke import digest
    folder = run_dir / "checkpoints/best-inference"
    folder.mkdir(parents=True, exist_ok=True)
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("D-095 inference serialization requires CPU model tensors")
    model.eval()
    model.save_pretrained(folder, safe_serialization=True)
    processor.save_pretrained(folder)
    atomic_json(folder / "training_adapter.json", adapter.manifest(model), replace=True)
    files = {path.name: digest(path) for path in folder.iterdir() if path.is_file()}
    manifest = {
        "schema_version": "bantai.stage5-soup.v1",
        "status": "complete",
        "authorization": "D-095",
        "resume_supported": False,
        "training_resume_supported": False,
        "purpose": "private_inference_replay_not_promoted",
        "parameter_sha256": parameter_sha256,
        "frozen_parameter_sha256": state_hash(model, True),
        "best_state": state,
        "recipe": recipe,
        "evaluation_sha256": digest(eval_path),
        "trainable_tensors_averaged": 558 if "soup" in state else 0,
        "files": files,
    }
    atomic_json(folder / "soup_manifest.json", manifest, replace=True)
    for name, expected in manifest["files"].items():
        if digest(folder / name) != expected:
            raise ValueError("D-095 inference bundle file hash mismatch")
    return folder, manifest


def train_step_d095(model, adapter, optimizer, batch, w_img: float, theta_0: list,
                    lambda_anchor: float = 0.004, lambda_pos: float = 1.5,
                    lambda_interior: float = 2.0, lambda_absent: float = 2.0):
    """
    Box-contrastive GT-region margin loss + interior wrong-class sub-box suppression +
    absent-class suppression + proximal L2 anchor.
    """
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    result = adapter(model, batch)
    components = checked_loss(result, model.config)
    detection_loss = result.loss

    prob_9 = _class_score_vectors(result.outputs, batch["input_ids"])  # [1, 900, 9]
    pred_boxes = result.outputs.pred_boxes[0]  # [900, 4] normalized cxcywh

    gt_labels = batch["labels"][0]
    gt_classes = gt_labels["class_labels"]  # [M]
    gt_boxes = gt_labels["boxes"]  # [M, 4] normalized cxcywh
    M = int(gt_classes.shape[0])

    device = detection_loss.device

    loss_pos_margin = torch.tensor(0.0, device=device)
    loss_interior = torch.tensor(0.0, device=device)

    if M > 0:
        q_cx, q_cy, q_w, q_h = pred_boxes.unbind(-1)
        q_x1 = (q_cx - 0.5 * q_w).unsqueeze(-1)  # [900, 1]
        q_y1 = (q_cy - 0.5 * q_h).unsqueeze(-1)
        q_x2 = (q_cx + 0.5 * q_w).unsqueeze(-1)
        q_y2 = (q_cy + 0.5 * q_h).unsqueeze(-1)

        g_cx, g_cy, g_w, g_h = gt_boxes.unbind(-1)
        g_x1 = (g_cx - 0.5 * g_w).unsqueeze(0)  # [1, M]
        g_y1 = (g_cy - 0.5 * g_h).unsqueeze(0)
        g_x2 = (g_cx + 0.5 * g_w).unsqueeze(0)
        g_y2 = (g_cy + 0.5 * g_h).unsqueeze(0)

        inter_x1 = torch.max(q_x1, g_x1)
        inter_y1 = torch.max(q_y1, g_y1)
        inter_x2 = torch.min(q_x2, g_x2)
        inter_y2 = torch.min(q_y2, g_y2)
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
        inter_area = inter_w * inter_h  # [900, M]

        q_area = torch.clamp((q_x2 - q_x1) * (q_y2 - q_y1), min=1e-6)  # [900, 1]
        g_area = torch.clamp((g_x2 - g_x1) * (g_y2 - g_y1), min=1e-6)  # [1, M]
        union_area = torch.clamp(q_area + g_area - inter_area, min=1e-6)

        iou = inter_area / union_area  # [900, M]
        iomin = inter_area / q_area   # [900, M]

        for m in range(M):
            cg = int(gt_classes[m])
            p_cg = prob_9[0, :, cg]
            match_score = iou[:, m].detach() + 0.5 * p_cg.detach()
            q_star = int(torch.argmax(match_score))
            p_qstar = prob_9[0, q_star]

            competing_mask = torch.ones(9, dtype=torch.bool, device=device)
            competing_mask[cg] = False
            max_competing = p_qstar[competing_mask].max()
            margin_diff = p_qstar[cg] - max_competing
            margin_penalty = torch.relu(0.25 - margin_diff).pow(2)
            if cg in (6, 7):  # Scale positive margin boost 2.0x for rare classes
                margin_penalty = margin_penalty * 2.0
            loss_pos_margin = loss_pos_margin + margin_penalty

            # Queries in region: IoU >= 0.30 or IoMin >= 0.75 and smaller than GT box
            in_region = (iou[:, m] >= 0.30) | ((iomin[:, m] >= 0.75) & (q_area[:, 0] < 0.85 * g_area[0, m]))
            region_indices = torch.where(in_region)[0]
            if len(region_indices) > 0:
                valid_classes = {cg}
                for m_other in range(M):
                    if m_other != m and (iou[region_indices, m_other] >= 0.40).any():
                        valid_classes.add(int(gt_classes[m_other]))
                wrong_classes = [c for c in range(9) if c not in valid_classes]
                if wrong_classes:
                    wrong_prob = prob_9[0, region_indices][:, wrong_classes]
                    loss_interior = loss_interior + torch.relu(wrong_prob - 0.05).pow(2).mean()

        loss_pos_margin = loss_pos_margin / max(1, M)
        loss_interior = loss_interior / max(1, M)

    # Auxiliary Absent-Class Suppression Loss (threshold 0.05 on top-15 queries)
    present_classes = set(int(c) for c in gt_classes)
    absent_classes = [c for c in range(9) if c not in present_classes]
    if absent_classes:
        loss_absent = sum(torch.relu(torch.topk(prob_9[0, :, c], k=15).values - 0.05).pow(2).mean()
                          for c in absent_classes) / len(absent_classes)
    else:
        loss_absent = torch.tensor(0.0, device=device)

    # Proximal L2 parameter anchor to theta_0
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    l2_anchor_loss = torch.tensor(0.0, device=device)
    for p, p0 in zip(trainable_params, theta_0):
        l2_anchor_loss = l2_anchor_loss + (p - p0).pow(2).sum()
    anchor_term = lambda_anchor * l2_anchor_loss

    total_loss = (w_img * detection_loss +
                  lambda_pos * loss_pos_margin +
                  lambda_interior * loss_interior +
                  lambda_absent * loss_absent +
                  anchor_term)
    total_loss.backward()

    probes = ('model.enc_output.weight', 'model.encoder_output_bbox_embed.layers.2.weight')
    named = dict(model.named_parameters())
    gradients = {}
    for name, p in named.items():
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
            raise ValueError('Missing/nonfinite training gradient: ' + name)
    for name in probes:
        gradients[name] = float(named[name].grad.norm())
        if gradients[name] <= 0:
            raise ValueError('Expected nonempty encoder gradient: ' + name)
    norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0, error_if_nonfinite=True)
    optimizer.step()

    return {
        "loss": float(total_loss.detach().cpu()),
        "detection_loss": float(detection_loss.detach().cpu()),
        "pos_margin_loss": float(loss_pos_margin.detach().cpu()),
        "interior_loss": float(loss_interior.detach().cpu()),
        "absent_loss": float(loss_absent.detach().cpu()),
        "anchor_loss": float(l2_anchor_loss.detach().cpu()),
        "w_img": float(w_img),
        "loss_ce": float(components["loss_ce"]),
        "loss_bbox": float(components["loss_bbox"]),
        "loss_giou": float(components["loss_giou"]),
        "clip_norm": float(norm.detach().cpu()) if hasattr(norm, "detach") else float(norm),
        "encoder_gradients": gradients,
        "seconds": time.perf_counter() - start,
    }


def run_stage5_d095_box_contrastive(run_dir: Path = TARGET_RUN) -> dict[str, object]:
    import torch
    start_all = time.perf_counter()
    run_dir = Path(run_dir).resolve()

    if (run_dir / "report.json").is_file():
        summary = strict_load(run_dir / "report.json")
        print(f"D-095 run already finished. Best state: {summary['best_state']} (AP={summary['best_AP']:.6f})", flush=True)
        return summary

    before = protected_snapshot(exclude_run=run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "integrity-before.json", before, replace=True)

    output = run_dir / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)

    d073 = strict_load(ROOT / "runs/training/stage5-data-expansion-audit-5412a13c0258467ba33a1f475cc4b65c/summary.json")
    export_manifest = strict_load(D095_EXPORT / "preparation_manifest.json")
    train_meta = strict_load(D095_EXPORT / "train.json")
    val_meta = strict_load(D095_EXPORT / "validation.json")

    if (len(train_meta["images"]), len(train_meta["annotations"])) != (576, 1240):
        raise ValueError(f"D-095 training export counts mismatch: {len(train_meta['images'])} images, {len(train_meta['annotations'])} boxes")
    if (len(val_meta["images"]), len(val_meta["annotations"])) != (65, 109):
        raise ValueError("D-095 validation export changed")

    d073_frozen = d073["frozen"]
    if set(i["id"] for i in val_meta["images"]) != set(d073_frozen["validation"]["ids"]):
        raise ValueError("D-095 validation IDs differ from the frozen 65-image set")

    dataset_root = Path(val_meta["info"]["image_root"]).resolve()
    permitted = {dataset_root / row["file_name"] for row in train_meta["images"] + val_meta["images"]}
    frozen_test = set(d073_frozen["test"]["ids"])
    raw = strict_load(export_manifest["source"]["annotations"])
    raw_by_id = {row["id"]: row for row in raw["images"]}
    test_paths = {(dataset_root / raw_by_id[ident]["file_name"]).resolve() for ident in frozen_test}
    if {Path(p).resolve() for p in permitted} & test_paths:
        raise ValueError("Training or validation photo overlaps frozen test IDs")

    _install_access_guard(dataset_root, permitted, run_dir)
    require_training_environment()

    train_ds = CocoTier3CoreDataset(D095_EXPORT, "train", D095_RIGHTS, mask_omitted=True)
    val_ds = CocoTier3CoreDataset(D095_EXPORT, "validation", D095_RIGHTS, mask_omitted=False)

    image_classes_map = defaultdict(list)
    for ann in train_ds.data["annotations"]:
        image_classes_map[ann["image_id"]].append(ann["category_id"])

    # Verify warmstart bundle
    warmstart_proof = _bundle_manifest_hashes(WARMSTART_BUNDLE)
    if warmstart_proof["parameter_sha256"] != EXPECTED_WARMSTART:
        raise ValueError(f"D-093 Champion parameter digest mismatch: {warmstart_proof['parameter_sha256']} vs {EXPECTED_WARMSTART}")

    model, processor, adapter = _runtime_model("cuda")
    warmstart = load_verified_stage5_checkpoint(model, WARMSTART_BUNDLE)
    if warmstart["parameter_sha256"] != EXPECTED_WARMSTART:
        raise ValueError("Warm-start did not load D-093 Champion exactly")
    model.to("cuda")

    frozen_hash = state_hash(model, True)
    if frozen_hash != FROZEN_BACKBONE_HASH:
        raise ValueError(f"Backbone frozen hash changed: {frozen_hash} vs {FROZEN_BACKBONE_HASH}")

    theta_0 = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    call_counts = {"training": 0, "validation": 0}

    def count_pre_hook(module, args, kwargs):
        mode = "training" if module.training and torch.is_grad_enabled() else "validation"
        limit = 1152 if mode == "training" else 560
        if call_counts[mode] >= limit:
            raise AssertionError(f"D-095 {mode} forward budget exceeded: {call_counts[mode]} >= {limit}")
        pixel_values = kwargs["pixel_values"]
        if tuple(pixel_values.shape) != (1, 3, 576, 576) or pixel_values.dtype != torch.float32:
            raise ValueError("D-095 384/576 FP32 forward geometry changed")
        call_counts[mode] += 1

    handle = model.register_forward_pre_hook(count_pre_hook, with_kwargs=True)

    contract = {
        "schema_version": "bantai.stage5-d095-box-contrastive-training.v1",
        "authorization": "D-095",
        "training_images": 576,
        "training_annotations": 1240,
        "omitted_accessory_boxes": 286,
        "validation_images": 65,
        "validation_annotations": 109,
        "shortest": 384,
        "longest": 576,
        "square_pad": 576,
        "epochs": 2,
        "batch_size": 1,
        "seed": 42,
        "optimizer": {
            "name": "AdamW",
            "lr_max": 1.5e-6,
            "lr_min": 2.0e-7,
            "schedule": "cosine_annealing",
            "weight_decay": 1e-4,
            "foreach": False,
        },
        "clip_norm": 1.0,
        "precision": "FP32",
        "TF32": False,
        "device": torch.cuda.get_device_name(0),
        "warmstart": warmstart,
        "frozen_parameter_sha256": frozen_hash,
        "rights_sha256": train_ds.rights["record_sha256"],
        "dataset_context": train_ds.context,
        "class_weights": {str(k): v for k, v in CLASS_WEIGHTS.items()},
        "lambda_l2_anchor": 0.004,
        "lambda_pos_margin": 1.5,
        "lambda_interior": 2.0,
        "lambda_absent": 2.0,
    }
    atomic_json(run_dir / "contract.json", contract, replace=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1.5e-6,
        weight_decay=1e-4,
        foreach=False,
    )

    epochs = 2
    steps_per_epoch = 576
    total_steps = epochs * steps_per_epoch  # 1152
    lr_max = 1.5e-6
    lr_min = 2.0e-7

    step_log_path = run_dir / "steps.jsonl"
    global_step = 0
    epoch_states = {}
    epoch_hashes = {}

    training_done = (output / "epoch-02").exists() and step_log_path.exists()
    if training_done:
        with step_log_path.open("r", encoding="utf-8") as stream:
            logged_steps = [json.loads(line) for line in stream]
        if len(logged_steps) == 1152:
            print("Detected completed 1,152-step training checkpoints. Loading epoch checkpoints for validation...", flush=True)
            call_counts["training"] = 1152
            model.to("cpu")
            for ep_num in (1, 2):
                ep_tag = f"epoch-0{ep_num}"
                load_verified_stage5_checkpoint(model, output / ep_tag)
                epoch_hashes[ep_num] = state_hash(model)
                epoch_states[ep_num] = {name: p.detach().cpu().clone()
                                        for name, p in model.named_parameters() if p.requires_grad}

    if not epoch_states:
        model.train()
        for ep in range(epochs):
            epoch_num = ep + 1
            batches = epoch_batches(576, 1, 42, ep)
            for batch_indices in batches:
                global_step += 1
                current_lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * global_step / total_steps))
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

                batch, geometry = get_batch(train_ds, batch_indices, processor, 384, 576)
                batch = move_batch(batch, "cuda")

                img_id = geometry[0]["image_id"]
                img_cats = image_classes_map[img_id]
                w_img = max(CLASS_WEIGHTS[c] for c in img_cats) if img_cats else 1.0

                metrics = train_step_d095(
                    model, adapter, optimizer, batch, w_img=w_img, theta_0=theta_0,
                    lambda_anchor=0.004, lambda_pos=1.5, lambda_interior=2.0, lambda_absent=2.0,
                )
                metrics["step"] = global_step
                metrics["epoch"] = epoch_num
                metrics["lr"] = current_lr

                with open(step_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics) + "\n")

                if global_step % 144 == 0 or global_step == total_steps:
                    print(f"[D-095] Step {global_step:04d}/{total_steps} (Epoch {epoch_num}/{epochs}) | "
                          f"loss={metrics['loss']:.4f} det={metrics['detection_loss']:.4f} "
                          f"pos={metrics['pos_margin_loss']:.4f} int={metrics['interior_loss']:.4f} "
                          f"abs={metrics['absent_loss']:.4f} lr={current_lr:.3e}", flush=True)

            # End of epoch checkpoint
            if state_hash(model, True) != frozen_hash:
                raise ValueError(f"Frozen parameters changed at end of epoch {epoch_num}")
            ep_tag = f"epoch-0{epoch_num}"
            epoch_position = {"epoch": epoch_num, "next_batch": 0, "global_step": global_step}
            save_checkpoint(output / ep_tag, model, processor, adapter, optimizer,
                            contract, epoch_position, {"frozen_parameter_sha256": frozen_hash})
            verify_checkpoint(output / ep_tag, contract)

            epoch_hashes[epoch_num] = state_hash(model)
            epoch_states[epoch_num] = {name: p.detach().cpu().clone()
                                       for name, p in model.named_parameters() if p.requires_grad}

    if call_counts["training"] != 1152 or state_hash(model, True) != frozen_hash:
        raise AssertionError("D-095 training forward count/frozen hash mismatch")

    del theta_0
    gc.collect()
    torch.cuda.empty_cache()

    # Load warmstart champion parameters for soup recipes
    warmstart_model, _, _ = _runtime_model("cpu")
    load_verified_stage5_checkpoint(warmstart_model, WARMSTART_BUNDLE)
    d093_champ_params = {name: p.detach().cpu().clone()
                         for name, p in warmstart_model.named_parameters() if p.requires_grad}
    del warmstart_model
    gc.collect()

    # Load D-093 E3 checkpoint parameters for hybrid soup
    d093_e3_model, _, _ = _runtime_model("cpu")
    load_verified_stage5_checkpoint(d093_e3_model, D093_E3_CHECKPOINT)
    d093_e3_params = {name: p.detach().cpu().clone()
                      for name, p in d093_e3_model.named_parameters() if p.requires_grad}
    del d093_e3_model
    gc.collect()

    # Load D-091 Champion parameters for hybrid soup
    d091_champ_model, _, _ = _runtime_model("cpu")
    load_verified_stage5_checkpoint(d091_champ_model, D091_CHAMP_BUNDLE)
    d091_champ_params = {name: p.detach().cpu().clone()
                         for name, p in d091_champ_model.named_parameters() if p.requires_grad}
    del d091_champ_model
    gc.collect()

    # 5-State Validation Evaluation (5 * 112 = 560 forwards)
    model.eval()
    validation_snapshot = rng_state()
    evaluations = {}
    state_vectors = {}

    # 1-2: Evaluate D095-BC E1 and E2
    epoch_results = {}
    for epoch_num in (1, 2):
        label = f"D095-BC E{epoch_num}"
        fname = f"validation-d095-bc-e{epoch_num}"
        print(f"Evaluating State {epoch_num}/5: {label}...", flush=True)

        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(epoch_states[epoch_num][name])

        param_hash = state_hash(model)
        if param_hash != epoch_hashes[epoch_num]:
            raise ValueError(f"State hash mismatch loading {label}: {param_hash} vs {epoch_hashes[epoch_num]}")
        model.to("cuda")

        prior_val = call_counts["validation"]
        try:
            result = evaluate_d095(model, processor, adapter, val_ds, "cuda", 384, 576,
                                   forward_counter=lambda: call_counts["validation"] - prior_val)
        finally:
            restore_rng(validation_snapshot)

        result["parameter_sha256"] = param_hash
        vecs, saved = _save_eval(run_dir, fname, result)
        evaluations[label] = saved
        state_vectors[label] = vecs
        epoch_results[epoch_num] = saved

        model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    # Determine best epoch state
    best_epoch_num = max((1, 2), key=lambda ep: epoch_results[ep]["combined_65"]["AP"])
    best_epoch_label = f"D095-BC E{best_epoch_num}"
    best_epoch_params = epoch_states[best_epoch_num]
    print(f"Best single-epoch state: {best_epoch_label} (AP={epoch_results[best_epoch_num]['combined_65']['AP']:.6f})", flush=True)

    # 3-5: Evaluate soup recipes
    recipes = (
        ("soup_d095best_d093champ_5050", {"D093_Champion": 0.50, "D095_Best": 0.50}),
        ("soup_d095best_d093champ_7525", {"D093_Champion": 0.25, "D095_Best": 0.75}),
        ("soup_d095e2_d093e3_d091_hybrid", {"D095_E2": 0.40, "D093_E3": 0.30, "D091_Champion": 0.30}),
    )

    soup_param_states = {}
    for idx, (label, recipe_weights) in enumerate(recipes, start=3):
        print(f"Evaluating State {idx}/5: {label} ({recipe_weights})...", flush=True)
        prior_val = call_counts["validation"]

        if label == "soup_d095e2_d093e3_d091_hybrid":
            averaged = average_trainable(
                {"d095": epoch_states[2], "d093": d093_e3_params, "d091": d091_champ_params},
                {"d095": 0.40, "d093": 0.30, "d091": 0.30},
            )
        else:
            d093_w = recipe_weights["D093_Champion"]
            d095_w = recipe_weights["D095_Best"]
            averaged = average_trainable(
                {"d093": d093_champ_params, "d095": best_epoch_params},
                {"d093": d093_w, "d095": d095_w},
            )

        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(averaged[name])

        param_hash = state_hash(model)
        if state_hash(model, True) != frozen_hash:
            raise ValueError("D-095 soup modified frozen parameters")
        model.to("cuda")

        try:
            result = evaluate_d095(model, processor, adapter, val_ds, "cuda", 384, 576,
                                   forward_counter=lambda: call_counts["validation"] - prior_val)
        finally:
            restore_rng(validation_snapshot)

        result["parameter_sha256"] = param_hash
        vecs, saved = _save_eval(run_dir, label, result)
        evaluations[label] = saved
        state_vectors[label] = vecs
        soup_param_states[label] = averaged

        model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    if call_counts["validation"] != 560:
        raise AssertionError(f"Expected exactly 560 validation forwards, performed {call_counts['validation']}")

    # Identify overall champion among all 5 states
    all_states = [
        "D095-BC E1", "D095-BC E2",
        "soup_d095best_d093champ_5050",
        "soup_d095best_d093champ_7525",
        "soup_d095e2_d093e3_d091_hybrid",
    ]
    best_state_name = max(all_states, key=lambda s: evaluations[s]["combined_65"]["AP"])
    best_eval = evaluations[best_state_name]
    best_param_hash = best_eval["parameter_sha256"]

    print(f"Overall Champion State: {best_state_name} (AP={best_eval['combined_65']['AP']:.6f})", flush=True)

    # Apply champion parameters to model for serialization
    if best_state_name in soup_param_states:
        champ_params = soup_param_states[best_state_name]
    elif best_state_name == "D095-BC E1":
        champ_params = epoch_states[1]
    else:
        champ_params = epoch_states[2]

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(champ_params[name])

    model.to("cpu")
    if state_hash(model) != best_param_hash:
        raise ValueError("Champion parameter serialization hash mismatch")

    if "soup" in best_state_name:
        best_eval_path = run_dir / f"{best_state_name}.json"
    else:
        best_eval_path = run_dir / f"validation-{best_state_name.lower().replace(' ', '-')}.json"
    bundle_dir, bundle_manifest = _save_inference_bundle(
        model, processor, adapter, run_dir, best_state_name, best_param_hash,
        best_eval_path, recipe={"state": best_state_name, "parameter_sha256": best_param_hash},
    )

    # Save best score vectors as best-score-vectors.npz
    best_vector_path = state_vectors[best_state_name]
    with np.load(best_vector_path) as best_npz:
        np.savez_compressed(run_dir / "best-score-vectors.npz",
                            image_ids=best_npz["image_ids"], scores=best_npz["scores"])

    # Run scene-level consensus audit
    val_data = strict_load(D095_EXPORT / "validation.json")
    with np.load(run_dir / "best-score-vectors.npz") as data:
        scores_by_img = {int(i): data["scores"][idx] for idx, i in enumerate(data["image_ids"])}
    champ_eval_data = strict_load(best_eval_path)
    consensus_audit = _run_scene_consensus_audit(val_data, champ_eval_data["raw_queries"], scores_by_img)
    atomic_json(run_dir / "d079-score-preserving-consensus.json", consensus_audit, replace=True)

    summary = {
        "authorization": "D-095",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "status": "complete",
        "device": torch.cuda.get_device_name(0),
        "total_seconds": time.perf_counter() - start_all,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "forward_counts": call_counts,
        "evaluations": {
            name: {
                "parameter_sha256": evaluations[name]["parameter_sha256"],
                "combined_65_AP": evaluations[name]["combined_65"]["AP"],
                "combined_65_AP50": evaluations[name]["combined_65"]["AP50"],
                "combined_65_AR100": evaluations[name]["combined_65"]["AR100"],
                "legacy_47_AP": evaluations[name]["legacy_47"]["AP"],
                "paper_AP": evaluations[name]["combined_65"]["per_class"]["paper"],
                "can_AP": evaluations[name]["combined_65"]["per_class"]["drink_can"],
                "styrofoam_AP": evaluations[name]["combined_65"]["per_class"]["styrofoam_piece"],
                "cup_AP": evaluations[name]["combined_65"]["per_class"]["plastic_cup"],
                "wrapper_AP": evaluations[name]["combined_65"]["per_class"]["plastic_wrapper"],
                "bottle_AP": evaluations[name]["combined_65"]["per_class"]["plastic_bottle"],
                "film_AP": evaluations[name]["combined_65"]["per_class"]["plastic_film"],
                "cigarette_AP": evaluations[name]["combined_65"]["per_class"]["cigarette"],
                "bantai_5_category_AP": evaluations[name]["material_hierarchy"]["raw_bantai_5_category"]["AP"],
            }
            for name in evaluations
        },
        "best_state": best_state_name,
        "best_AP": evaluations[best_state_name]["combined_65"]["AP"],
        "best_parameter_sha256": bundle_manifest["parameter_sha256"],
        "consensus_audit": {
            "changed_queries": consensus_audit["changed_queries"],
            "suppressed_queries": consensus_audit["suppressed_queries"],
            "raw_9_specific_AP": consensus_audit["baseline"]["9-specific"]["metrics"]["AP"],
            "consensus_9_specific_AP": consensus_audit["consensus"]["9-specific"]["metrics"]["AP"],
            "raw_bantai_5_category_AP": consensus_audit["baseline"]["bantai-5-category"]["metrics"]["AP"],
            "consensus_bantai_5_category_AP": consensus_audit["consensus"]["bantai-5-category"]["metrics"]["AP"],
        },
    }
    atomic_json(run_dir / "report.json", summary, replace=True)
    atomic_json(run_dir / "training-run-state.json", {"phase": "complete", "status": "complete", "summary": summary}, replace=True)
    after_proof = verify_snapshot(before, exclude_run=run_dir)
    atomic_json(run_dir / "integrity-after-training.json", after_proof, replace=True)

    print(f"D-095 complete. Best state: {best_state_name} (AP={summary['best_AP']:.6f})", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=TARGET_RUN)
    args = parser.parse_args()
    run_stage5_d095_box_contrastive(args.run_dir)
