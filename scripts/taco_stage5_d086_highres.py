"""Bounded Stage 5 High-Resolution (448/672) 1-Epoch Fine-Tuning & Evaluation (D-086).

Executes:
1. 1-epoch high-res training at 448/672 (AdamW lr=1.0e-6, weight_decay=1e-4, FP32, TF32 off).
2. 4-state validation evaluation (448 validation forwards total = 4 states x 112 forwards):
   - D086-HR448 E1 @ 448/672
   - soup_d086_d083_5050_hr448 @ 448/672
   - soup_d086_d083_5050_std384 @ 384/576
   - soup_d086_d083_2575_std384 @ 384/576
3. Score vector persistence and best-inference saving under no-regression rule.
4. CPU-only BANTAI 5-Category, D-079 consensus, and Multi-Scale Query Fusion Audit.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
from scripts.train_taco import SOURCES, train_step
from scripts.train_taco_stage5 import freeze_backbones, load_verified_stage5_checkpoint
from scripts.taco_weight_integration import average_trainable
from scripts.taco_stage5_material_hierarchy import evaluate_case, OBJECT_REGISTRY
from scripts.taco_stage5_d083_core import (
    protected_snapshot, verify_snapshot, _bundle_manifest_hashes, _runtime_model,
    save_vectors, _consensus_audit, _install_access_guard, _save_inference_bundle,
    EXPECTED_D074, evaluate_d083
)
from scripts.taco_error_analysis import iou

EXPECTED_D083_CHAMPION = "82c0a5bf3a104db5f5ff35e1cbfc1bd239595ec9946e22cf085986985d95fddc"
EXPECTED_D083_E1 = "662961b424417dd4050c3dcbebb0feab0bb6a2b88421a0ced1fa1a3525b9d24b"
FROZEN_BACKBONE_HASH = "161c9814bffe1657ae0debd7e21b7ed1c41f29f3ba43fffe879ab31ba36a0b67"

D083_RUN = ROOT / "runs/training/stage5-d083-core-20261031-a"
D083_CHAMPION_BUNDLE = D083_RUN / "checkpoints/best-inference"
D083_E1_BUNDLE = D083_RUN / "checkpoints/epoch-01"
D074_BUNDLE = ROOT / "runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337/checkpoints/soup-tier1-hr-blend2575"
D083_EXPORT = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1"
D083_RIGHTS = ROOT / "work/stage5-paper/rights_d083_core.json"
LEGACY_JSON = ROOT / "exports/taco/strict-clean-stage2-v1/validation.json"

DEFAULT_D086_RUN = ROOT / "runs/training/stage5-d086-highres-20261031-a"


def evaluate_state_multiscale(model, processor, adapter, dataset, device="cuda",
                              shortest=448, longest=672, forward_counter=None):
    """Evaluate 112 validation forwards (65 combined + 47 legacy) at (shortest, longest)."""
    res = evaluate_d083(model, processor, adapter, dataset, device, shortest=shortest, longest=longest,
                        forward_counter=forward_counter)
    res["shortest"] = shortest
    res["longest"] = longest
    return res


def _save_eval_meta(run_dir, state, result):
    vectors = result.pop("class_vectors")
    meta = save_vectors(run_dir / f"scores-{state}.npz", vectors)
    result["score_vectors"] = meta
    atomic_json(run_dir / f"{state}.json", result, replace=True)
    return vectors, result


def _multi_scale_query_fusion(val_data, std_queries, hr_queries):
    """CPU-only query fusion combining 384/576 and 448/672 queries."""
    from scripts.taco_stage5_material_hierarchy import prepare_rows, evaluate_case, SCHEMES

    by_img = defaultdict(lambda: {"std": [], "hr": []})
    for r in std_queries:
        by_img[r["image_id"]]["std"].append(r)
    for r in hr_queries:
        by_img[r["image_id"]]["hr"].append(r)

    fused_pool = []
    # Class-aware IoU 0.55 NMS across combined candidate pool
    for img_id, pools in by_img.items():
        combined = pools["std"] + pools["hr"]
        sorted_rows = sorted(combined, key=lambda r: -r["score"])
        kept = []
        for r in sorted_rows:
            suppress = False
            for k in kept:
                if k["category_id"] != r["category_id"]:
                    continue
                if iou(k["bbox"], r["bbox"]) >= 0.55:
                    suppress = True
                    break
            if not suppress:
                kept.append(r)
        fused_pool.extend(sorted(kept, key=lambda r: -r["score"])[:100])

    # Evaluate fused pool on 9-specific and BANTAI 5-category
    nine_spec = evaluate_case(val_data, fused_pool, "9-specific", "raw")
    bantai5 = evaluate_case(val_data, fused_pool, "bantai-5-category", "raw")

    return {
        "status": "complete",
        "method": "Class-aware IoU 0.55 NMS on std384 + hr448 queries, top-100 per image",
        "fused_query_count": len(fused_pool),
        "9_specific": nine_spec["metrics"],
        "bantai_5_category": bantai5["metrics"]
    }


def _train_step_highres_cpu_opt(model, adapter, optimizer, cpu_params, trainable_params_gpu, batch):
    """Custom high-res training step with CPU optimizer offload to fit 6GB VRAM budget."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    result = adapter(model, batch)
    components = checked_loss(result, model.config)
    result.loss.backward()

    named = dict(model.named_parameters())
    for name, p in named.items():
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
            raise ValueError("Missing/nonfinite training gradient: " + name)

    for p_gpu, p_cpu in zip(trainable_params_gpu, cpu_params):
        p_cpu.grad = p_gpu.grad.detach().cpu()
        p_gpu.grad = None

    norm = torch.nn.utils.clip_grad_norm_(cpu_params, 1.0, error_if_nonfinite=True)
    optimizer.step()

    for p_gpu, p_cpu in zip(trainable_params_gpu, cpu_params):
        p_gpu.data.copy_(p_cpu.data)

    loss = float(result.loss.detach())
    del result
    return {
        "loss": loss,
        "components": components,
        "gradient_norm_before_clip": float(norm),
        "seconds": time.perf_counter() - start
    }


def train_and_evaluate_d086(run_dir: Path):
    run_dir = Path(run_dir).resolve()
    before = protected_snapshot(exclude_run=run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "integrity-before.json", before, replace=True)

    d083_champ_proof = _bundle_manifest_hashes(D083_CHAMPION_BUNDLE)
    d074_proof = _bundle_manifest_hashes(D074_BUNDLE)
    if d083_champ_proof["parameter_sha256"] != EXPECTED_D083_CHAMPION:
        raise ValueError(f"D083 champion parameter hash mismatch: {d083_champ_proof['parameter_sha256']}")
    if d074_proof["parameter_sha256"] != EXPECTED_D074:
        raise ValueError(f"D074 bundle parameter hash mismatch: {d074_proof['parameter_sha256']}")

    d073 = strict_load(ROOT / "runs/training/stage5-data-expansion-audit-5412a13c0258467ba33a1f475cc4b65c/summary.json")
    val_meta = strict_load(D083_EXPORT / "validation.json")
    train_meta = strict_load(D083_EXPORT / "train.json")
    dataset_root = Path(val_meta["info"]["image_root"]).resolve()
    permitted = {dataset_root / row["file_name"] for row in train_meta["images"] + val_meta["images"]}
    _install_access_guard(dataset_root, permitted, run_dir)
    require_training_environment()

    train_ds = CocoTrainingDatasetV2(D083_EXPORT, "train", D083_RIGHTS)
    val_ds = CocoTrainingDatasetV2(D083_EXPORT, "validation", D083_RIGHTS)
    if (len(train_ds), len(train_ds.data["annotations"])) != (379, 731):
        raise ValueError("D-086 loader dataset counts changed")
    if (len(val_ds), len(val_ds.data["annotations"])) != (65, 109):
        raise ValueError("D-086 loader validation counts changed")

    model, processor, adapter = _runtime_model("cuda")
    warmstart = load_verified_stage5_checkpoint(model, D083_CHAMPION_BUNDLE)
    if warmstart["parameter_sha256"] != EXPECTED_D083_CHAMPION:
        raise ValueError("Warmstart did not load D083 champion exactly")

    freeze_backbones(model)
    model.to("cuda")
    frozen_hash = state_hash(model, True)
    if frozen_hash != FROZEN_BACKBONE_HASH:
        raise ValueError(f"Frozen backbone hash mismatch: {frozen_hash} != {FROZEN_BACKBONE_HASH}")

    call_counts = {"training": 0, "validation": 0}

    def count_pre_hook(module, args, kwargs):
        mode = "training" if module.training and torch.is_grad_enabled() else "validation"
        limit = 379 if mode == "training" else 448
        if call_counts[mode] >= limit:
            raise AssertionError(f"D-086 {mode} forward budget exceeded: {call_counts[mode]} >= {limit}")
        pixel_values = kwargs["pixel_values"]
        if mode == "training":
            if tuple(pixel_values.shape) != (1, 3, 672, 672) or pixel_values.dtype != torch.float32:
                raise ValueError("D-086 448/672 FP32 training geometry mismatch")
        call_counts[mode] += 1

    handle = model.register_forward_pre_hook(count_pre_hook, with_kwargs=True)

    contract = {
        "authorization": "D-086",
        "run_id": run_dir.name,
        "labels": list(LABELS),
        "prompt": PROMPT,
        "training_images": 379,
        "training_annotations": 731,
        "validation_images": 65,
        "validation_annotations": 109,
        "shortest": 448,
        "longest": 672,
        "square_pad": 672,
        "epochs": 1,
        "batch_size": 1,
        "seed": 42,
        "optimizer": {"name": "AdamW", "lr": 1.0e-6, "weight_decay": 1e-4, "foreach": False, "offload": "cpu"},
        "clip_norm": 1.0,
        "precision": "FP32",
        "TF32": False,
        "device": torch.cuda.get_device_name(0),
        "warmstart": warmstart,
        "frozen_parameter_sha256": frozen_hash,
        "checkpoint_every_steps": 100,
        "test_access": False
    }
    atomic_json(run_dir / "contract.json", contract, replace=True)
    atomic_json(run_dir / "training-run-state.json", {"phase": "training", "status": "running", "contract": contract}, replace=True)

    trainable_params_gpu = [p for p in model.parameters() if p.requires_grad]
    cpu_params = [p.detach().cpu().clone().requires_grad_(True) for p in trainable_params_gpu]
    optimizer = torch.optim.AdamW(cpu_params, lr=1.0e-6, weight_decay=1e-4, foreach=False)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    steps = []
    batches = epoch_batches(379, 1, 42, 0)
    output = run_dir / "checkpoints"
    output.mkdir(exist_ok=True)

    model.train()
    for step, batch_indexes in enumerate(batches, start=1):
        index = batch_indexes[0]
        batch, geometry = get_batch(train_ds, [index], processor, 448, 672)
        batch = move_batch(batch, "cuda")
        row = _train_step_highres_cpu_opt(model, adapter, optimizer, cpu_params, trainable_params_gpu, batch)
        row.update(step=step, batch_index=index, image_id=train_ds.images[index]["id"])
        steps.append(row)

        with (run_dir / "steps.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        del batch
        torch.cuda.empty_cache()

        # VRAM Safety Gate on step 1
        if step == 1:
            peak_step1 = torch.cuda.max_memory_allocated()
            if peak_step1 > 5800 * 2**20:
                raise MemoryError(f"Step 1 VRAM allocation {peak_step1 / 2**20:.1f} MiB exceeded 5800 MiB safety gate")

        if step % 50 == 0 or step == 379:
            print(f"D-086 HR update {step}/379 loss={row['loss']:.5f} peak={torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)

        if step % 100 == 0:
            if state_hash(model, True) != frozen_hash:
                raise ValueError("Frozen parameters changed before checkpoint")
            position = {"epoch": 0, "next_batch": step, "global_step": step}
            save_checkpoint(output / f"step-{step:06d}", model, processor, adapter, optimizer,
                            contract, position, {"frozen_parameter_sha256": frozen_hash})
            verify_checkpoint(output / f"step-{step:06d}", contract)

    epoch_position = {"epoch": 1, "next_batch": 0, "global_step": 379}
    save_checkpoint(output / "epoch-01", model, processor, adapter, optimizer,
                    contract, epoch_position, {"frozen_parameter_sha256": frozen_hash})
    verify_checkpoint(output / "epoch-01", contract)

    e1_hash = state_hash(model)
    e1_state = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}

    # Evaluate State 1: D086-HR448 E1 @ 448/672
    model.eval()
    val_snap = rng_state()
    try:
        res_e1 = evaluate_state_multiscale(model, processor, adapter, val_ds, "cuda", 448, 672)
    finally:
        restore_rng(val_snap)
    res_e1["parameter_sha256"] = e1_hash
    e1_vectors, saved_e1 = _save_eval_meta(run_dir, "validation-d086-hr448-e1", res_e1)

    evaluations = {"D086-HR448 E1": saved_e1}
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    del optimizer
    model.to("cpu")
    gc.collect(); torch.cuda.empty_cache()

    # Load D083 champion trainable weights for blending
    proof_champ = load_verified_stage5_checkpoint(model, D083_CHAMPION_BUNDLE)
    champ_state = {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}

    # Blend recipes:
    # 2. soup_d086_d083_5050_hr448 @ 448/672
    # 3. soup_d086_d083_5050_std384 @ 384/576
    # 4. soup_d086_d083_2575_std384 @ 384/576
    soup_5050 = average_trainable({"champ": champ_state, "hr_e1": e1_state}, {"champ": 0.5, "hr_e1": 0.5})
    soup_2575 = average_trainable({"champ": champ_state, "hr_e1": e1_state}, {"champ": 0.75, "hr_e1": 0.25})

    candidate_runs = [
        ("soup_d086_d083_5050_hr448", soup_5050, 448, 672),
        ("soup_d086_d083_5050_std384", soup_5050, 384, 576),
        ("soup_d086_d083_2575_std384", soup_2575, 384, 576),
    ]

    for label, trainable_dict, short_val, long_val in candidate_runs:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(trainable_dict[name])
        p_hash = state_hash(model)
        model.to("cuda")
        val_snap = rng_state()
        try:
            res_cand = evaluate_state_multiscale(model, processor, adapter, val_ds, "cuda", short_val, long_val)
        finally:
            restore_rng(val_snap)
        res_cand["parameter_sha256"] = p_hash
        cand_vectors, saved_cand = _save_eval_meta(run_dir, label, res_cand)
        evaluations[label] = saved_cand
        model.to("cpu")
        gc.collect(); torch.cuda.empty_cache()

    handle.remove()
    if call_counts != {"training": 379, "validation": 448}:
        raise AssertionError(f"D-086 forward count mismatch: {call_counts}")

    # Determine best state in D-086 by combined_65 AP
    best_state_name = max(evaluations, key=lambda k: evaluations[k]["combined_65"]["AP"])
    best_eval = evaluations[best_state_name]
    best_ap = best_eval["combined_65"]["AP"]

    # No-regression promotion check against D-083 Champion (0.236129)
    d083_champ_ap = 0.2361288920465962
    promoted_as_repo_default = best_ap > d083_champ_ap

    # Serialize best bundle for D-086 run
    best_trainable = (e1_state if best_state_name == "D086-HR448 E1"
                      else (soup_5050 if "5050" in best_state_name else soup_2575))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(best_trainable[name])
    bundle_dir, bundle_man = _save_inference_bundle(
        model, processor, adapter, run_dir, best_state_name, best_eval["parameter_sha256"],
        run_dir / f"{best_state_name}.json" if (run_dir / f"{best_state_name}.json").is_file() else run_dir / "validation-d086-hr448-e1.json",
        {"recipe": best_state_name}
    )
    best_vec_dict = {img["id"]: arr for img, arr in zip(val_ds.images, np.load(run_dir / f"scores-{'validation-d086-hr448-e1' if best_state_name == 'D086-HR448 E1' else best_state_name}.npz")["scores"])}
    best_vec_meta = save_vectors(run_dir / "best-score-vectors.npz", best_vec_dict)

    from scripts.taco_stage5_d083_core import _evaluate_materials
    rows_by_state = {name: saved["raw_queries"] for name, saved in evaluations.items()}
    val_data = val_ds.data
    material_audit = _evaluate_materials(val_data, rows_by_state)
    atomic_json(run_dir / "material-hierarchy-audit.json", material_audit, replace=True)

    consensus = _consensus_audit(val_data, best_eval["raw_queries"], best_vec_dict)
    atomic_json(run_dir / "d079-score-preserving-consensus.json", consensus, replace=True)

    # Multi-Scale Query Fusion Audit: combine D-083 champion queries (384) with D-086 HR queries (448)
    d083_champ_data = strict_load(D083_RUN / "soup_d083_d074_5050.json")
    fusion_audit = _multi_scale_query_fusion(val_data, d083_champ_data["raw_queries"], saved_e1["raw_queries"])
    atomic_json(run_dir / "multiscale-fusion-audit.json", fusion_audit, replace=True)

    peak_vram = torch.cuda.max_memory_allocated()
    report = {
        "authorization": "D-086",
        "status": "complete",
        "run_id": run_dir.name,
        "contract": contract,
        "training_updates": 379,
        "training_forwards": call_counts["training"],
        "validation_forwards": call_counts["validation"],
        "training_mean_loss": sum(r["loss"] for r in steps) / len(steps),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_VRAM_bytes": peak_vram,
        "warmstart_parameter_sha256": warmstart["parameter_sha256"],
        "epoch_parameter_sha256": e1_hash,
        "states": {
            name: {
                "parameter_sha256": val["parameter_sha256"],
                "combined_65": val["combined_65"],
                "legacy_47": val["legacy_47"],
                "mean_loss_per_image": val["mean_loss_per_image"],
                "validation_forwards": val["validation_forwards"],
                "shortest": val["shortest"],
                "longest": val["longest"]
            }
            for name, val in evaluations.items()
        },
        "best_state": {
            "state": best_state_name,
            "AP": best_ap,
            "parameter_sha256": best_eval["parameter_sha256"],
            "promoted_as_repo_default": promoted_as_repo_default,
            "d083_champ_reference_ap": d083_champ_ap
        },
        "best_bundle": str(bundle_dir),
        "best_bundle_parameter_sha256": bundle_man["parameter_sha256"],
        "material_hierarchy_audit": "material-hierarchy-audit.json",
        "d079_consensus_audit": "d079-score-preserving-consensus.json",
        "multiscale_fusion_audit": "multiscale-fusion-audit.json",
        "frozen_parameters_unchanged": state_hash(model, True) == frozen_hash,
        "test_json_opened": False,
        "heldout_test_photos_opened": 0
    }
    atomic_json(run_dir / "report.json", report, replace=True)
    after = verify_snapshot(before, run_dir)
    atomic_json(run_dir / "integrity-after-training.json", after, replace=True)
    atomic_json(run_dir / "training-run-state.json", {
        "phase": "complete",
        "forward_counts": call_counts,
        "best_state": best_state_name,
        "promoted_as_repo_default": promoted_as_repo_default,
        "protected_artifacts_unchanged": True
    }, replace=True)

    print(json.dumps({
        "run_dir": str(run_dir),
        "best_state": best_state_name,
        "best_AP": best_ap,
        "promoted_as_repo_default": promoted_as_repo_default,
        "peak_vram_mib": peak_vram / 2**20,
        "multiscale_fusion_9spec_AP": fusion_audit["9_specific"]["AP"],
        "multiscale_fusion_bantai5_AP": fusion_audit["bantai_5_category"]["AP"]
    }, indent=2))
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_D086_RUN)
    args = parser.parse_args()
    train_and_evaluate_d086(args.run_dir)
