#!/usr/bin/env python3
"""
Stage 5 Trainer: Nine-class Grounding DINO fine-tuning with 'paper' (Option A).

Warm-starts from champion Stage 4 weights (blend_350_442_2575) or pretrained snapshot.
Trains on 307 strict-clean images (221 frozen original + 86 new paper images).
Evaluates on all 65 validation images, logging:
  - combined_65: All 65 images across 9 classes
  - legacy_47: Original 47 holdouts across 8 classes (direct comparison with Stage 4)
  - annex_18: 18 new holdouts across 9 classes (paper evaluation)
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid

import torch
from safetensors import safe_open
from transformers import AutoProcessor, GroundingDinoForObjectDetection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.configuration import MODEL_ID, REVISION, strict_load
from scripts.persistence import atomic_json
from scripts.training_smoke import (
    SNAPSHOT,
    checked_loss,
    digest,
    loading_ok,
    move_batch,
    require_training_environment,
    state_hash,
)
from scripts.taco_training_labels_v2 import LABELS, PROMPT, validate_taxonomy
from scripts.training_adapter_v2 import EncoderOutputAdapterV2, adapter_source_digest
from scripts.taco_training_data_v2 import CocoTrainingDatasetV2, get_batch
from scripts.taco_evaluation_v3 import evaluate_stage5
from scripts.taco_weight_integration import average_trainable, canonical_saved_keys, bitwise_equal
from scripts.train_taco import train_step, SOURCES
from scripts.training_checkpoint import save_checkpoint, verify_checkpoint, epoch_batches, rng_state, restore_rng


def freeze_backbones(model):
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith(("model.backbone.", "model.text_backbone.")))


def load_warmstart_blend(model, run_dir: Path):
    cp_350 = run_dir / "checkpoints/step-000350"
    cp_442 = run_dir / "checkpoints/epoch-02"
    if not cp_350.is_dir() or not cp_442.is_dir():
        raise FileNotFoundError(f"Missing Stage 4 checkpoints in {run_dir}")

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    weights = {350: 0.25, 442: 0.75}
    cps = {350: cp_350, 442: cp_442}

    states = {350: {}, 442: {}}
    context=strict_load(run_dir/"report.json")["contract"]
    persisted=set(model.state_dict())
    base=dict(model.named_parameters());base.update({n:t for n,t in model.named_buffers() if n in persisted})
    if len(trainable)!=558:raise ValueError("Expected original558 trainable tensors")
    for step, cp in cps.items():
        verify_checkpoint(cp,context)
        with safe_open(str(cp / "model.safetensors"), framework="pt", device="cpu") as f:
            mapping = canonical_saved_keys(model, f.keys())
            for saved_name, key in mapping.items():
                if key in trainable:
                    states[step][key] = f.get_tensor(saved_name)
                elif not bitwise_equal(f.get_tensor(saved_name),base[key]):
                    raise ValueError("Warmstart frozen entry differs: "+key)

    averaged = average_trainable(states, weights)
    with torch.no_grad():
        named=dict(model.named_parameters())
        for name,tensor in averaged.items():named[name].copy_(tensor)
    proof=strict_load(ROOT/"runs/training/scale576-integration-c70249cde2f2422cb89b22232fd2b525/record.json")["states"]["blend_350_442_2575"]
    if state_hash(model)!=proof["parameter_sha256"]:raise ValueError("Warmstart does not reproduce the D061 champion hash")
    print("Successfully warm-started model with champion blend_350_442_2575 weights.")


def load_stage5_blend2575(model, run_dir: Path):
    cp1 = run_dir / "checkpoints/epoch-01"
    cp2 = run_dir / "checkpoints/epoch-02"
    if not cp1.is_dir() or not cp2.is_dir():
        raise FileNotFoundError(f"Missing Stage 5 checkpoints in {run_dir}")

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    weights = {"ep1": 0.25, "ep2": 0.75}
    states = {"ep1": {}, "ep2": {}}

    for name, cp in [("ep1", cp1), ("ep2", cp2)]:
        with safe_open(str(cp / "model.safetensors"), framework="pt", device="cpu") as f:
            mapping = canonical_saved_keys(model, f.keys())
            for saved_name, key in mapping.items():
                if key in trainable:
                    states[name][key] = f.get_tensor(saved_name)

    averaged = average_trainable(states, weights)
    with torch.no_grad():
        named = dict(model.named_parameters())
        for name, tensor in averaged.items():
            named[name].copy_(tensor)

    expected_hash = "5a8b832754cfbf16fa56e4742299c61a8f60cdba6cba9d6c9758ec85ad7b3e98"
    if state_hash(model) != expected_hash:
        raise ValueError("Warmstart does not match verified Stage 5 blend_2575 hash")
    print(f"Successfully warm-started model with Stage 5 blend_2575 weights ({len(averaged)} tensors).")


def save_stage5_checkpoint(output_dir: Path, step_name: str, model, optimizer, epoch: int, step: int, contract: dict,processor,adapter):
    cp_dir = output_dir / "checkpoints" / step_name
    count=contract["training_images"]
    save_checkpoint(cp_dir,model,processor,adapter,optimizer,contract,
        {"epoch":step//count,"next_batch":step%count,"global_step":step},
        {"frozen_parameter_sha256":state_hash(model,True)})
    verify_checkpoint(cp_dir,contract)
    pointer=output_dir/"latest_checkpoint.json"
    atomic_json(pointer,{"latest":step_name,"path":str(cp_dir)},replace=pointer.exists())
    return cp_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stage 5 9-class Waste Detection Trainer")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (default: 2)")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--warmstart", default="blend", choices=["blend", "snapshot", "stage5_blend2575"], help="Warmstart source")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--dataset", default="exports/taco/strict-clean-stage5-v1", help="Dataset export directory")
    parser.add_argument("--rights", default="work/stage5-paper/rights.json", help="Rights declaration file")
    parser.add_argument("--shortest", type=int, default=256, help="Shortest canvas edge")
    parser.add_argument("--longest", type=int, default=384, help="Longest canvas edge")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--evaluate-baseline", action="store_true", help="Run validation before training")
    args = parser.parse_args(argv)
    if not 1<=args.epochs<=3 or not math.isfinite(args.learning_rate) or args.learning_rate<=0:
        parser.error("Require1..3 epochs and finite positive LR")
    if (args.shortest,args.longest) not in ((256,384),(384,576)):
        parser.error("Qualified input scales are256/384 or384/576")

    require_training_environment()
    validate_taxonomy()
    torch.set_num_threads(4)

    device = args.device
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    run_id = f"stage5-paper-{uuid.uuid4().hex}"
    output_dir = ROOT / "runs" / "training" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"=== Stage 5 9-Class Training Run: {run_id} ===")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device} | LR: {args.learning_rate} | Epochs: {args.epochs} | Canvas: {args.shortest}/{args.longest}")

    # Load datasets
    print("Loading Stage 5 training & validation datasets...")
    train_ds = CocoTrainingDatasetV2(args.dataset, "train", args.rights)
    val_ds = CocoTrainingDatasetV2(args.dataset, "validation", args.rights)
    print(f"  Training samples: {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    if len(train_ds)!=307 or len(val_ds)!=65:raise ValueError("Require the approved307/65 corpus")
    # Persist the exact write/read boundary before model execution. All source
    # images/checkpoints remain read-only; test metadata/photos are unavailable.
    train_photos={(train_ds.root/i["file_name"]).resolve() for i in train_ds.images}
    val_photos={(val_ds.root/i["file_name"]).resolve() for i in val_ds.images}
    if train_photos&val_photos:raise ValueError("Train/validation photograph overlap")
    def file_guard(event,values):
        if event=="open" and isinstance(values[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(values[0])).resolve()
            if path.name=="test.json":raise AssertionError("Test split access prohibited")
            if path.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp") and path.is_relative_to(train_ds.root):
                if path not in train_photos|val_photos:raise AssertionError("Test/non-authorized photo access prohibited")
            flags=values[2] if len(values)>2 else 0
            if isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
                if path.is_relative_to(ROOT) and not path.is_relative_to(output_dir):raise AssertionError("Existing workspace artifacts are read-only")
    sys.addaudithook(file_guard)

    # Load model and processor
    print(f"Loading Grounding DINO Swin-T model from {SNAPSHOT}...")
    model, info = GroundingDinoForObjectDetection.from_pretrained(
        SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        disable_custom_kernels=True,
        output_loading_info=True,
    )
    loading_ok(info)
    processor = AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    freeze_backbones(model)

    if args.warmstart == "blend":
        stage4_run = ROOT / "runs/training/taco-896e2f44146d4d54aee518e6087908ca"
        load_warmstart_blend(model, stage4_run)
    elif args.warmstart == "stage5_blend2575":
        stage5_run = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4"
        load_stage5_blend2575(model, stage5_run)

    model.to(device)
    adapter = EncoderOutputAdapterV2(model, processor.tokenizer)
    calls={"training":0,"validation":0}
    def forward_guard(module,args_hook,kwargs):
        mode="training" if module.training and torch.is_grad_enabled() else "validation"
        limit=len(train_ds)*args.epochs if mode=="training" else 112*(args.epochs+int(args.evaluate_baseline))
        if calls[mode]>=limit:raise ValueError("Model forward budget exceeded")
        if list(kwargs['pixel_values'].shape)!=[1,3,args.longest,args.longest] or kwargs['pixel_values'].dtype!=torch.float32:
            raise ValueError("Image geometry/FP32 contract changed")
        calls[mode]+=1
    model.register_forward_pre_hook(forward_guard,with_kwargs=True)

    # Build contract
    contract = {
        "run_id": run_id,
        "labels": list(LABELS),
        "prompt": PROMPT,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "warmstart": args.warmstart,
        "shortest": args.shortest,
        "longest": args.longest,
        "seed": args.seed,
        "device": device,
        "gpu_name": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "dataset_context": train_ds.context,
        "training_images":len(train_ds),"validation_images":len(val_ds),
        "implementation_sha256":{name:digest(ROOT/name) for name in SOURCES},
        "stage5_source_sha256":{name:digest(ROOT/"scripts"/name) for name in
            ("train_taco_stage5.py","training_adapter_v2.py","taco_training_data_v2.py","taco_training_labels_v2.py","taco_evaluation_v3.py")},
        "rights_sha256":train_ds.rights["record_sha256"],
        "optimizer":{"name":"AdamW","lr":args.learning_rate,"weight_decay":1e-4,"foreach":False},
        "precision":"FP32","TF32":False,"clip_norm":1.0,"sampler":"epoch_batches(n,1,seed,zero_based_epoch)",
    }
    atomic_json(output_dir/"contract.json",contract)
    atomic_json(output_dir/"run-state.json",{"status":"running","contract":contract,"forward_counts":calls})
    frozen_hash=state_hash(model,True)
    def validation():
        snapshot=rng_state();weights=state_hash(model)
        try:
            with torch.inference_mode():value=evaluate_stage5(model,processor,adapter,val_ds,device,args.shortest,args.longest)
        finally:restore_rng(snapshot)
        if state_hash(model)!=weights or state_hash(model,True)!=frozen_hash:raise ValueError("Validation changed parameters")
        return value

    # Baseline evaluation if requested
    baseline_metrics = None
    if args.evaluate_baseline:
        print("\nEvaluating baseline across 65 validation images...")
        baseline_metrics = validation()
        print(f"Baseline Results:")
        print(f"  Combined 65 AP: {baseline_metrics['combined_65']['AP']:.6f} | AP50: {baseline_metrics['combined_65']['AP50']:.6f}")
        print(f"  Legacy 47   AP: {baseline_metrics['legacy_47']['AP']:.6f} | AP50: {baseline_metrics['legacy_47']['AP50']:.6f}")
        print(f"  Annex 18    AP: {baseline_metrics['annex_18']['AP']:.6f} | AP50: {baseline_metrics['annex_18']['AP50']:.6f}")
        atomic_json(output_dir / "baseline.json", baseline_metrics)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-4,
        foreach=False,
    )

    # Training loop
    print(f"\nStarting bounded training ({args.epochs} epochs)...")
    global_step = 0
    epoch_results = []
    shuffled_indices = list(range(len(train_ds)))

    start_train_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        shuffled_indices=[v[0] for v in epoch_batches(len(train_ds),1,args.seed,epoch-1)]

        for step_in_epoch, img_idx in enumerate(shuffled_indices, 1):
            global_step += 1
            batch, _ = get_batch(train_ds, [img_idx], processor, args.shortest, args.longest)
            batch = move_batch(batch, device)

            row=train_step(model,adapter,optimizer,batch)
            del batch
            loss_val=row["loss"]
            with (output_dir/"steps.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(dict(row,global_step=global_step,epoch=epoch,batch_index=step_in_epoch-1,image_id=train_ds.images[img_idx]["id"]))+"\n")
            epoch_losses.append(loss_val)

            if global_step % 50 == 0 or global_step == len(train_ds) * args.epochs:
                vram_str = f" | VRAM: {torch.cuda.max_memory_allocated() / (1024 * 1024):.0f}MB" if device == "cuda" else ""
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step_in_epoch}/{len(train_ds)}] Global {global_step} | Loss: {loss_val:.4f}{vram_str}")

            if global_step % 100 == 0:
                if state_hash(model,True)!=frozen_hash:raise ValueError("Frozen parameters changed")
                save_stage5_checkpoint(output_dir, f"step-{global_step:06d}", model, optimizer, epoch, global_step, contract,processor,adapter)

        # Epoch checkpoint
        if state_hash(model,True)!=frozen_hash:raise ValueError("Frozen parameters changed")
        save_stage5_checkpoint(output_dir, f"epoch-{epoch:02d}", model, optimizer, epoch, global_step, contract,processor,adapter)

        # Validation at epoch end
        print(f"\n--- Validation after Epoch {epoch} ---")
        val_eval = validation()
        print(f"Epoch {epoch} Evaluation:")
        print(f"  Combined 65 AP: {val_eval['combined_65']['AP']:.6f} | AP50: {val_eval['combined_65']['AP50']:.6f}")
        print(f"  Legacy 47   AP: {val_eval['legacy_47']['AP']:.6f} | AP50: {val_eval['legacy_47']['AP50']:.6f}")
        print(f"  Annex 18    AP: {val_eval['annex_18']['AP']:.6f} | AP50: {val_eval['annex_18']['AP50']:.6f}")
        print(f"  Paper AP (Annex): {val_eval['annex_18']['classes']['paper']['AP']}")
        atomic_json(output_dir / f"validation-epoch-{epoch}.json", val_eval)

        epoch_results.append({
            "epoch": epoch,
            "mean_loss": sum(epoch_losses) / len(epoch_losses),
            "combined_ap": val_eval["combined_65"]["AP"],
            "combined_ap50": val_eval["combined_65"]["AP50"],
            "legacy_ap": val_eval["legacy_47"]["AP"],
            "legacy_ap50": val_eval["legacy_47"]["AP50"],
            "annex_ap": val_eval["annex_18"]["AP"],
            "annex_ap50": val_eval["annex_18"]["AP50"],
            "paper_ap": val_eval["annex_18"]["classes"]["paper"]["AP"],
        })
        atomic_json(output_dir/"run-state.json",{"status":"running","contract":contract,"completed_epochs":epoch_results,
            "global_step":global_step,"forward_counts":calls},replace=True)

    total_time = time.time() - start_train_time
    report = {
        "run_id": run_id,
        "contract": contract,
        "baseline": baseline_metrics,
        "epoch_results": epoch_results,
        "total_training_time_seconds": total_time,
        "completed": True,
        "training_updates":global_step,"training_photo_reads":train_ds.photo_reads,"validation_photo_reads":val_ds.photo_reads,
        "validation_forwards":calls["validation"],"training_forwards":calls["training"],
        "frozen_parameters_unchanged":state_hash(model,True)==frozen_hash,
        "core_sources_unchanged":{name:digest(ROOT/name) for name in SOURCES}==contract["implementation_sha256"],
    }
    if calls!={"training":len(train_ds)*args.epochs,"validation":112*(args.epochs+int(args.evaluate_baseline))}:
        raise ValueError("Completed forward counts differ from the authorized epoch/evaluation plan")
    atomic_json(output_dir / "report.json", report)
    atomic_json(output_dir/"run-state.json",{"status":"completed","global_step":global_step,"forward_counts":calls},replace=True)
    print(f"\nTraining completed in {total_time:.1f}s. Report saved to {output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
