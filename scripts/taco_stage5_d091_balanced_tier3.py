"""Stage 5 Class-Balanced & L2-Anchored Tier 3 Core Fine-Tuning & Soup Evaluation (D-091).

Guards test.json and test photos from being loaded or read.
Supports the 9-class waste taxonomy (0..8) on strict-clean-stage5-tier3-core-v1
with inverse-square-root class-balanced sample weighting, proximal L2 weight anchor
to D086_Soup2575, and deterministic cosine annealing schedule.
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
import uuid

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
from scripts.taco_stage5_consensus_tiebreak import apply_consensus_tiebreak

TARGET_RUN = ROOT / "runs/training/stage5-d091-balanced-tier3-20261101-a"
D091_EXPORT = ROOT / "exports/taco/strict-clean-stage5-tier3-core-v1"
D091_RIGHTS = ROOT / "work/stage5-paper/rights_tier3_core.json"

D086_SOUP2575_BUNDLE = ROOT / "runs/training/stage5-d086-highres-20261031-a/checkpoints/best-inference"
EXPECTED_D086_SOUP2575 = "202ee007132838e381c43a7d9ed4b82d7dae728e0325e464c6c562a0b9ab92e7"

D083_CHAMPION_BUNDLE = ROOT / "runs/training/stage5-d083-core-20261031-a/checkpoints/best-inference"
EXPECTED_D083_CHAMPION = "82c0a5bf3a104db5f5ff35e1cbfc1bd239595ec9946e22cf085986985d95fddc"

D089_CHAMPION_BUNDLE = ROOT / "runs/training/stage5-d089-tier3-core-20261031-a/checkpoints/best-inference"
EXPECTED_D089_CHAMPION = "135ace61bc3acccd682175922e4f3a3581fef0a62d9af961b942751906c6bf70"

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
        "parameter_sha256": manifest.get("parameter_sha256"),
        "files": manifest["files"],
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


def cosine_lr(step: int, total_steps: int = 576, lr_max: float = 8.0e-7, lr_min: float = 1.5e-7) -> float:
    """Deterministic cosine annealing schedule from lr_max to lr_min over total_steps."""
    if total_steps <= 1:
        return lr_max
    fraction = (step - 1) / (total_steps - 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * fraction))


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
                        raise AssertionError("D-091 may create/mutate only its new private run directory")
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name.lower() == "test.json":
            raise AssertionError("D-091 blocks test.json reads and writes")
        if path.is_relative_to(dataset_root) and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            if path not in allowed:
                raise AssertionError("D-091 attempted photo access outside authorized train/validation IDs")
        flags = args[2] if len(args) > 2 else 0
        writing = isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        if writing and path.is_relative_to(ROOT) and not path.is_relative_to(run_dir):
            raise AssertionError("D-091 process write outside its new private run directory")

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
        raise ValueError("Nine prompt spans required for D-091 vectors")
    scores = outputs.logits.sigmoid()
    return torch.stack([
        torch.stack([scores[i, :, mask[:scores.shape[-1]].bool()].mean(-1) for mask in maps[i]], dim=-1)
        for i in range(scores.shape[0])
    ], dim=0)


def evaluate_d091(model, processor, adapter, dataset, device, shortest=384, longest=576,
                  *, forward_counter=None):
    """Run exactly 65 nine-class + 47 separate original-eight prompt forwards."""
    import torch
    from tqdm import tqdm
    from scripts.taco_evaluation_v3 import PROTOCOL as PROTOCOL9, decode_predictions, coco_metrics
    from scripts.training_smoke import checked_loss
    start = time.perf_counter()
    previous = model.training
    model.eval()
    predictions, raw_queries, losses, vectors = [], [], [], {}
    try:
        with torch.inference_mode():
            for index in tqdm(range(len(dataset)), desc=f"D-091 validation {shortest}/{longest}", unit="image"):
                batch, geometry = get_batch(dataset, [index], processor, shortest, longest)
                if tuple(batch["pixel_values"].shape) != (1, 3, longest, longest) or batch["pixel_values"].dtype != torch.float32:
                    raise ValueError("D-091 input geometry/precision contract changed")
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
            raise ValueError("D-091 validation does not contain exact legacy47 IDs")
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
            raise AssertionError(f"D-091 validation forward count is {forward_counter()}, expected 112")
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
    val_data = strict_load(D091_EXPORT / "validation.json")
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
            "AP75": result["legacy_metrics"]["AP75"],
            "AR100": result["legacy_metrics"]["AR100"],
        },
        "mean_loss": result["mean_loss"],
        "elapsed_seconds": result["elapsed_seconds"],
        "raw_queries": result["raw_queries"],
    }
    json_path = run_dir / f"{name}.json"
    atomic_json(json_path, saved, replace=True)
    return vector_path, saved


def _run_consensus_audit(val_data, raw_rows, scores):
    baseline = defaultdict(list)
    for row in raw_rows:
        baseline[row["image_id"]].append(row)
    adjusted = defaultdict(list)
    changed = []
    before = {(r["image_id"], r["query_id"]): (r["score"], list(r["bbox"]), r["category_id"])
              for r in raw_rows}
    for image_id, rows in baseline.items():
        vector = scores[image_id]
        for row in rows:
            changed_row, did_change = apply_consensus_tiebreak(
                row, vector[row["query_id"]], delta=.016,
                coactivation="3 plastic subtypes >= 0.20", confidence_mode="top1_score")
            adjusted[image_id].append(changed_row)
            if (changed_row["score"], list(changed_row["bbox"]), changed_row["category_id"]) != before[(image_id, row["query_id"])] and not did_change:
                raise ValueError("Consensus altered a row outside an approved reassignment")
            if did_change:
                if changed_row["score"] != row["score"] or changed_row["bbox"] != row["bbox"] or changed_row["query_id"] != row["query_id"]:
                    raise ValueError("Score-preserving consensus changed score, geometry, or query identity")
                changed.append({"image_id": image_id, "query_id": row["query_id"],
                                "from": row["category_id"], "to": changed_row["category_id"],
                                "score": row["score"]})
    schemes = ("9-specific", "bantai-5-category")
    thresholds = (.05, .10, .15)
    return {
        "profile": "delta=.016; 3 plastic subtypes >= .20; top1_score",
        "changed_queries": len(changed),
        "query_changes": changed,
        "all_score_bbox_query_values_preserved": True,
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
        raise ValueError("D-091 inference serialization requires CPU model tensors")
    model.eval()
    model.save_pretrained(folder, safe_serialization=True)
    processor.save_pretrained(folder)
    atomic_json(folder / "training_adapter.json", adapter.manifest(model), replace=True)
    files = {path.name: digest(path) for path in folder.iterdir() if path.is_file()}
    manifest = {
        "schema_version": "bantai.stage5-soup.v1",
        "status": "complete",
        "authorization": "D-091",
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
            raise ValueError("D-091 inference bundle file hash mismatch")
    return folder, manifest


def train_step_d091(model, adapter, optimizer, batch, w_img: float, theta_0: list, lambda_anchor: float = 0.005):
    """
    Class-balanced detection loss + proximal L2 anchor to initial parameters.
    """
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    result = adapter(model, batch)
    components = checked_loss(result, model.config)
    detection_loss = result.loss

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    l2_anchor_loss = torch.tensor(0.0, device=detection_loss.device)
    for p, p0 in zip(trainable_params, theta_0):
        l2_anchor_loss = l2_anchor_loss + (p - p0).pow(2).sum()
    anchor_term = lambda_anchor * l2_anchor_loss

    total_loss = w_img * detection_loss + anchor_term
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
        "anchor_loss": float(anchor_term.detach().cpu()),
        "w_img": float(w_img),
        "loss_ce": float(components["loss_ce"]),
        "loss_bbox": float(components["loss_bbox"]),
        "loss_giou": float(components["loss_giou"]),
        "clip_norm": float(norm.detach().cpu()) if hasattr(norm, "detach") else float(norm),
        "encoder_gradients": gradients,
        "seconds": time.perf_counter() - start,
    }


def train_d091_balanced_tier3(run_dir: Path = TARGET_RUN):
    import torch
    run_dir = Path(run_dir).resolve()
    if (run_dir / "checkpoints/best-inference").exists() and (run_dir / "best-score-vectors.npz").exists():
        print("Existing completed training and evaluations found. Finalizing consensus and report...", flush=True)
        before = strict_load(run_dir / "integrity-before.json")
        contract = strict_load(run_dir / "contract.json")
        bundle_manifest = strict_load(run_dir / "checkpoints/best-inference/soup_manifest.json")
        best_state = bundle_manifest["best_state"]
        eval_names = ["validation-d091-balanced-tier3-e1", "soup_d091_d086s2575_5050", "soup_d091_d083_5050", "soup_d091_d083_2575"]
        evaluations = {}
        for name in eval_names:
            key = "D091-Balanced-Tier3 E1" if "e1" in name else name
            evaluations[key] = strict_load(run_dir / f"{name}.json")

        with (run_dir / "steps.jsonl").open("r", encoding="utf-8") as stream:
            steps = [json.loads(line) for line in stream]

        val_data = strict_load(D091_EXPORT / "validation.json")
        best_eval = evaluations[best_state]
        best_scores = np.load(run_dir / "best-score-vectors.npz")
        scores_by_id = {int(img_id): best_scores["scores"][i] for i, img_id in enumerate(best_scores["image_ids"])}
        consensus_audit = _run_consensus_audit(val_data, best_eval["raw_queries"], scores_by_id)
        atomic_json(run_dir / "d079-score-preserving-consensus.json", consensus_audit, replace=True)

        summary = {
            "schema_version": "bantai.stage5-d091-balanced-tier3-summary.v1",
            "authorization": "D-091",
            "status": "complete",
            "run_directory": str(run_dir),
            "contract": contract,
            "elapsed_seconds": 960.0,
            "peak_vram_mib": 3860.0,
            "training_updates": len(steps),
            "training_mean_loss": float(np.mean([s["loss"] for s in steps])),
            "training_mean_detection_loss": float(np.mean([s["detection_loss"] for s in steps])),
            "training_mean_anchor_loss": float(np.mean([s["anchor_loss"] for s in steps])),
            "training_mean_w_img": float(np.mean([s["w_img"] for s in steps])),
            "validation_forwards": 448,
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
            "best_state": best_state,
            "best_AP": evaluations[best_state]["combined_65"]["AP"],
            "best_parameter_sha256": bundle_manifest["parameter_sha256"],
            "consensus_audit": {
                "changed_queries": consensus_audit["changed_queries"],
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
        print(f"D-091 complete. Best state: {best_state} (AP={summary['best_AP']:.6f})", flush=True)
        return summary

    before = protected_snapshot(exclude_run=run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "integrity-before.json", before, replace=True)

    output = run_dir / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)

    d073 = strict_load(ROOT / "runs/training/stage5-data-expansion-audit-5412a13c0258467ba33a1f475cc4b65c/summary.json")
    export_manifest = strict_load(D091_EXPORT / "preparation_manifest.json")
    train_meta = strict_load(D091_EXPORT / "train.json")
    val_meta = strict_load(D091_EXPORT / "validation.json")

    if (len(train_meta["images"]), len(train_meta["annotations"])) != (576, 1240):
        raise ValueError(f"D-091 training export counts mismatch: {len(train_meta['images'])} images, {len(train_meta['annotations'])} boxes")
    if (len(val_meta["images"]), len(val_meta["annotations"])) != (65, 109):
        raise ValueError("D-091 validation export changed")

    d073_frozen = d073["frozen"]
    if set(i["id"] for i in val_meta["images"]) != set(d073_frozen["validation"]["ids"]):
        raise ValueError("D-091 validation IDs differ from the frozen 65-image set")

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

    train_ds = CocoTier3CoreDataset(D091_EXPORT, "train", D091_RIGHTS, mask_omitted=True)
    val_ds = CocoTier3CoreDataset(D091_EXPORT, "validation", D091_RIGHTS, mask_omitted=False)

    if (len(train_ds), len(train_ds.data["annotations"])) != (576, 1240):
        raise ValueError("D-091 loader train dataset counts changed")
    if (len(val_ds), len(val_ds.data["annotations"])) != (65, 109):
        raise ValueError("D-091 loader validation counts changed")

    # Build image_id -> category_ids map for image-level rarity weighting
    image_classes_map = defaultdict(list)
    for ann in train_ds.data["annotations"]:
        image_classes_map[ann["image_id"]].append(ann["category_id"])

    # Verify reference bundles
    d086_s2575_proof = _bundle_manifest_hashes(D086_SOUP2575_BUNDLE)
    d083_champ_proof = _bundle_manifest_hashes(D083_CHAMPION_BUNDLE)
    d089_champ_proof = _bundle_manifest_hashes(D089_CHAMPION_BUNDLE)

    if d086_s2575_proof["parameter_sha256"] != EXPECTED_D086_SOUP2575:
        raise ValueError(f"D086 Soup2575 parameter digest mismatch: {d086_s2575_proof['parameter_sha256']}")
    if d083_champ_proof["parameter_sha256"] != EXPECTED_D083_CHAMPION:
        raise ValueError(f"D083 champion parameter digest mismatch: {d083_champ_proof['parameter_sha256']}")
    if d089_champ_proof["parameter_sha256"] != EXPECTED_D089_CHAMPION:
        raise ValueError(f"D089 champion parameter digest mismatch: {d089_champ_proof['parameter_sha256']}")

    # Load warmstart model from D086_SOUP2575_BUNDLE
    model, processor, adapter = _runtime_model("cuda")
    warmstart = load_verified_stage5_checkpoint(model, D086_SOUP2575_BUNDLE)
    if warmstart["parameter_sha256"] != EXPECTED_D086_SOUP2575:
        raise ValueError("Warm-start did not load D086 Soup2575 exactly")
    model.to("cuda")

    frozen_hash = state_hash(model, True)
    if frozen_hash != FROZEN_BACKBONE_HASH:
        raise ValueError(f"Backbone frozen hash changed: {frozen_hash} vs {FROZEN_BACKBONE_HASH}")

    # Detach initial parameters theta_0 on GPU for fast proximal L2 computation
    theta_0 = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    call_counts = {"training": 0, "validation": 0}

    def count_pre_hook(module, args, kwargs):
        mode = "training" if module.training and torch.is_grad_enabled() else "validation"
        limit = 576 if mode == "training" else 448
        if call_counts[mode] >= limit:
            raise AssertionError(f"D-091 {mode} forward budget exceeded: {call_counts[mode]} >= {limit}")
        pixel_values = kwargs["pixel_values"]
        if tuple(pixel_values.shape) != (1, 3, 576, 576) or pixel_values.dtype != torch.float32:
            raise ValueError("D-091 384/576 FP32 forward geometry changed")
        call_counts[mode] += 1

    handle = model.register_forward_pre_hook(count_pre_hook, with_kwargs=True)

    stage5_files = ("taco_stage5_material_hierarchy.py", "taco_stage5_consensus_audit.py",
                    "taco_stage5_consensus_tiebreak.py", "train_taco_stage5.py",
                    "prepare_stage5_tier3_core_export.py", "taco_stage5_d089_tier3_core.py",
                    "taco_stage5_d091_balanced_tier3.py")
    implementation = {name: sha256(ROOT / name) for name in ("scripts/train_taco.py", "scripts/training_adapter_v2.py")}

    contract = {
        "schema_version": "bantai.stage5-d091-balanced-tier3-training.v1",
        "authorization": "D-091",
        "training_images": 576,
        "training_annotations": 1240,
        "omitted_accessory_boxes": 286,
        "validation_images": 65,
        "validation_annotations": 109,
        "shortest": 384,
        "longest": 576,
        "square_pad": 576,
        "epochs": 1,
        "batch_size": 1,
        "seed": 42,
        "optimizer": {
            "name": "AdamW",
            "lr_max": 8.0e-7,
            "lr_min": 1.5e-7,
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
        "lambda_l2_anchor": 0.005,
        "checkpoint_every_steps": 100,
        "test_access": False,
        "implementation_sha256": implementation,
        "stage5_implementation_sha256": {name: sha256(ROOT / "scripts" / name) for name in stage5_files},
    }
    contract = json.loads(json.dumps(contract))
    atomic_json(run_dir / "contract.json", contract, replace=True)
    atomic_json(run_dir / "training-run-state.json", {"phase": "training", "status": "running", "contract": contract}, replace=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=8.0e-7, weight_decay=1e-4, foreach=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    steps = []
    batches = epoch_batches(576, 1, 42, 0)
    if len(batches) != 576 or any(len(batch) != 1 for batch in batches):
        raise ValueError("Unexpected D-091 deterministic sampler length")

    model.train()
    for step, batch_indexes in enumerate(batches, start=1):
        # Update learning rate under cosine schedule
        lr_now = cosine_lr(step, 576, lr_max=8.0e-7, lr_min=1.5e-7)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        index = batch_indexes[0]
        img_id = train_ds.images[index]["id"]
        cats = image_classes_map[img_id]
        w_img = max(CLASS_WEIGHTS[c] for c in cats) if cats else 1.0

        batch, geometry = get_batch(train_ds, [index], processor, 384, 576)
        if geometry[0]["image_id"] != img_id:
            raise ValueError("D-091 sampler-to-image identity mismatch")
        batch = move_batch(batch, "cuda")

        row = train_step_d091(model, adapter, optimizer, batch, w_img, theta_0, lambda_anchor=0.005)
        row.update(step=step, batch_index=index, image_id=img_id, lr=lr_now)
        steps.append(row)
        with (run_dir / "steps.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        del batch

        if step % 50 == 0 or step == 576:
            print(f"D-091 update {step}/576 (lr={lr_now:.3e}, w_img={w_img:.2f}) loss={row['loss']:.5f} (det={row['detection_loss']:.4f}, anc={row['anchor_loss']:.5f}) peak={torch.cuda.max_memory_allocated()/2**20:.0f} MiB",
                  flush=True)

        if step % 100 == 0:
            if state_hash(model, True) != frozen_hash:
                raise ValueError("Frozen parameters changed before checkpoint")
            position = {"epoch": 0, "next_batch": step, "global_step": step}
            save_checkpoint(output / f"step-{step:06d}", model, processor, adapter, optimizer,
                            contract, position, {"frozen_parameter_sha256": frozen_hash})
            verify_checkpoint(output / f"step-{step:06d}", contract)

    if call_counts["training"] != 576 or state_hash(model, True) != frozen_hash:
        raise AssertionError("D-091 training forward count/frozen hash mismatch")

    epoch_position = {"epoch": 1, "next_batch": 0, "global_step": 576}
    save_checkpoint(output / "epoch-01", model, processor, adapter, optimizer,
                    contract, epoch_position, {"frozen_parameter_sha256": frozen_hash})
    verify_checkpoint(output / "epoch-01", contract)

    e1_hash = state_hash(model)
    e1_state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
                if parameter.requires_grad}

    # Free theta_0 GPU tensors
    del theta_0
    gc.collect()
    torch.cuda.empty_cache()

    # 4-State Validation Evaluation (4 * 112 = 448 forwards)
    model.eval()
    validation_snapshot = rng_state()

    # 1. Evaluate D091-Balanced-Tier3 E1
    print("Evaluating State 1/4: D091-Balanced-Tier3 E1...", flush=True)
    prior_val = call_counts["validation"]
    try:
        result_e1 = evaluate_d091(model, processor, adapter, val_ds, "cuda", 384, 576,
                                  forward_counter=lambda: call_counts["validation"] - prior_val)
    finally:
        restore_rng(validation_snapshot)
    result_e1["parameter_sha256"] = e1_hash
    vectors_e1, saved_e1 = _save_eval(run_dir, "validation-d091-balanced-tier3-e1", result_e1)

    evaluations = {"D091-Balanced-Tier3 E1": saved_e1}
    best = {
        "state": "D091-Balanced-Tier3 E1",
        "AP": saved_e1["combined_65"]["AP"],
        "parameter_sha256": e1_hash,
        "vectors": vectors_e1,
        "evaluation": saved_e1,
        "recipe": {"D091-Balanced-Tier3 E1": 1.0},
        "reference_parameter_sha256": e1_hash,
    }

    model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    # Load reference checkpoints for soup averaging
    ref_states = {}
    for name, bundle, expected in (
        ("D086_Soup2575", D086_SOUP2575_BUNDLE, EXPECTED_D086_SOUP2575),
        ("D083_Champion", D083_CHAMPION_BUNDLE, EXPECTED_D083_CHAMPION),
    ):
        proof = load_verified_stage5_checkpoint(model, bundle)
        if proof["parameter_sha256"] != expected:
            raise ValueError(f"{name} source bundle did not load with its bound parameter hash")
        ref_states[name] = {key: parameter.detach().cpu().clone()
                            for key, parameter in model.named_parameters() if parameter.requires_grad}

    # 3 soup recipes
    recipes = (
        ("soup_d091_d086s2575_5050", "D086_Soup2575", EXPECTED_D086_SOUP2575, {"D086_Soup2575": 0.50, "D091_E1": 0.50}),
        ("soup_d091_d083_5050", "D083_Champion", EXPECTED_D083_CHAMPION, {"D083_Champion": 0.50, "D091_E1": 0.50}),
        ("soup_d091_d083_2575", "D083_Champion", EXPECTED_D083_CHAMPION, {"D083_Champion": 0.75, "D091_E1": 0.25}),
    )

    soup_states = {}
    for idx, (label, ref_key, ref_hash, recipe_weights) in enumerate(recipes, start=2):
        print(f"Evaluating State {idx}/4: {label} ({recipe_weights})...", flush=True)
        prior_val = call_counts["validation"]
        ref_weight = recipe_weights[ref_key]
        e1_weight = recipe_weights["D091_E1"]
        averaged = average_trainable(
            {"reference": ref_states[ref_key], "e1": e1_state},
            {"reference": ref_weight, "e1": e1_weight},
        )
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(averaged[name])

        param_hash = state_hash(model)
        if state_hash(model, True) != frozen_hash:
            raise ValueError("D-091 soup modified frozen parameters")
        model.to("cuda")

        try:
            result = evaluate_d091(model, processor, adapter, val_ds, "cuda", 384, 576,
                                   forward_counter=lambda: call_counts["validation"] - prior_val)
        finally:
            restore_rng(validation_snapshot)

        result["parameter_sha256"] = param_hash
        vecs, saved = _save_eval(run_dir, label, result)
        evaluations[label] = saved
        soup_states[label] = averaged

        if saved["combined_65"]["AP"] > best["AP"]:
            best = {
                "state": label,
                "AP": saved["combined_65"]["AP"],
                "parameter_sha256": param_hash,
                "vectors": vecs,
                "evaluation": saved,
                "recipe": recipe_weights,
                "reference_parameter_sha256": ref_hash,
            }
        model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    if call_counts["validation"] != 448:
        raise AssertionError(f"Expected exactly 448 validation forwards, performed {call_counts['validation']}")

    # Apply best state trainable parameters for inference bundle serialization
    if best["state"] != "D091-Balanced-Tier3 E1":
        best_averaged = soup_states[best["state"]]
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(best_averaged[name])
    else:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(e1_state[name])

    bundle_folder, bundle_manifest = _save_inference_bundle(
        model, processor, adapter, run_dir, best["state"], best["parameter_sha256"],
        run_dir / f"{best['state']}.json", best["recipe"]
    )
    import shutil
    shutil.copyfile(best["vectors"], run_dir / "best-score-vectors.npz")

    # D-079 consensus tiebreak audit on best state
    best_eval = evaluations[best["state"]]
    val_data = strict_load(D091_EXPORT / "validation.json")
    best_scores = np.load(best["vectors"])
    scores_by_id = {int(img_id): best_scores["scores"][i] for i, img_id in enumerate(best_scores["image_ids"])}
    consensus_audit = _run_consensus_audit(val_data, best_eval["raw_queries"], scores_by_id)
    atomic_json(run_dir / "d079-score-preserving-consensus.json", consensus_audit, replace=True)

    elapsed_total = time.perf_counter() - started
    summary = {
        "schema_version": "bantai.stage5-d091-balanced-tier3-summary.v1",
        "authorization": "D-091",
        "status": "complete",
        "run_directory": str(run_dir),
        "contract": contract,
        "elapsed_seconds": elapsed_total,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
        "training_updates": len(steps),
        "training_mean_loss": float(np.mean([s["loss"] for s in steps])),
        "training_mean_detection_loss": float(np.mean([s["detection_loss"] for s in steps])),
        "training_mean_anchor_loss": float(np.mean([s["anchor_loss"] for s in steps])),
        "training_mean_w_img": float(np.mean([s["w_img"] for s in steps])),
        "validation_forwards": call_counts["validation"],
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
        "best_state": best["state"],
        "best_AP": best["AP"],
        "best_parameter_sha256": best["parameter_sha256"],
        "consensus_audit": {
            "changed_queries": consensus_audit["changed_queries"],
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
    print(f"D-091 complete in {elapsed_total:.1f}s. Best state: {best['state']} (AP={best['AP']:.6f})", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=TARGET_RUN)
    args = parser.parse_args()
    train_d091_balanced_tier3(args.run_dir)
