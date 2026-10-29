#!/usr/bin/env python3
"""
Evaluate Stage 5 checkpoints at 384/576 scale and test checkpoint weight blending.

Computes:
  - combined_65 AP / AP50
  - legacy_47 AP / AP50
  - annex_18 AP / AP50 (including Paper AP)
Zero test set access; strictly operates on validation.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from transformers import AutoProcessor, GroundingDinoForObjectDetection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.configuration import MODEL_ID, REVISION, strict_load
from scripts.persistence import atomic_json
from scripts.training_smoke import (
    SNAPSHOT,
    loading_ok,
    require_training_environment,
)
from scripts.taco_training_labels_v2 import LABELS, validate_taxonomy
from scripts.training_adapter_v2 import EncoderOutputAdapterV2
from scripts.taco_training_data_v2 import CocoTrainingDatasetV2
from scripts.taco_evaluation_v3 import evaluate_stage5
from scripts.taco_weight_integration import average_trainable, canonical_saved_keys


def freeze_backbones(model):
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith(("model.backbone.", "model.text_backbone.")))


def load_checkpoint_weights(model, cp_dir: Path):
    safetensors_path = cp_dir / "model.safetensors"
    if not safetensors_path.is_file():
        raise FileNotFoundError(f"model.safetensors not found in {cp_dir}")

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {}
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        mapping = canonical_saved_keys(model, f.keys())
        for saved_name, key in mapping.items():
            if key in trainable:
                state[key] = f.get_tensor(saved_name)

    model.load_state_dict(state, strict=False)
    print(f"Loaded weights from {cp_dir} ({len(state)} trainable tensors)")


def load_blended_weights(model, cp_dirs: dict[str, Path], weights: dict[str, float]):
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    states = {}
    for name, cp_dir in cp_dirs.items():
        states[name] = {}
        with safe_open(str(cp_dir / "model.safetensors"), framework="pt", device="cpu") as f:
            mapping = canonical_saved_keys(model, f.keys())
            for saved_name, key in mapping.items():
                if key in trainable:
                    states[name][key] = f.get_tensor(saved_name)

    averaged = average_trainable(states, weights)
    model.load_state_dict(averaged, strict=False)
    print(f"Loaded blended weights {weights} across {list(cp_dirs.keys())}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Stage 5 models at scale")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4",
        help="Stage 5 run directory",
    )
    parser.add_argument("--checkpoint", default="epoch-02", help="Checkpoint name or 'blend_1_2'")
    parser.add_argument("--shortest", type=int, default=384, help="Shortest canvas edge (default: 384)")
    parser.add_argument("--longest", type=int, default=576, help="Longest canvas edge (default: 576)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset", default="exports/taco/strict-clean-stage5-v1")
    parser.add_argument("--rights", default="work/stage5-paper/rights.json")
    args = parser.parse_args()

    require_training_environment()
    validate_taxonomy()

    device = args.device
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"=== Stage 5 Evaluation at Scale {args.shortest}/{args.longest} ===")
    print(f"Target Checkpoint: {args.checkpoint} | Device: {device}")

    # Load dataset
    val_ds = CocoTrainingDatasetV2(args.dataset, "validation", args.rights)
    print(f"Validation dataset: {len(val_ds)} images loaded.")

    # Load base architecture
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

    # Load weights
    if args.checkpoint == "blend_1_2":
        cp1 = args.run_dir / "checkpoints/epoch-01"
        cp2 = args.run_dir / "checkpoints/epoch-02"
        load_blended_weights(model, {"ep1": cp1, "ep2": cp2}, {"ep1": 0.5, "ep2": 0.5})
    elif args.checkpoint == "blend_2575":
        cp1 = args.run_dir / "checkpoints/epoch-01"
        cp2 = args.run_dir / "checkpoints/epoch-02"
        load_blended_weights(model, {"ep1": cp1, "ep2": cp2}, {"ep1": 0.25, "ep2": 0.75})
    else:
        cp_dir = args.run_dir / "checkpoints" / args.checkpoint
        load_checkpoint_weights(model, cp_dir)

    model.to(device)
    adapter = EncoderOutputAdapterV2(model, processor.tokenizer)

    # Run evaluation
    print(f"\nRunning 65 validation forwards at {args.shortest}/{args.longest} canvas scale...")
    results = evaluate_stage5(model, processor, adapter, val_ds, device, args.shortest, args.longest)

    print("\n" + "=" * 60)
    print(f"RESULTS @ {args.shortest}/{args.longest} Canvas Scale:")
    print("=" * 60)
    print(f"Combined 65 AP:   {results['combined_65']['AP']:.6f} | AP50: {results['combined_65']['AP50']:.6f}")
    print(f"Legacy 47   AP:   {results['legacy_47']['AP']:.6f} | AP50: {results['legacy_47']['AP50']:.6f}")
    print(f"Annex 18    AP:   {results['annex_18']['AP']:.6f} | AP50: {results['annex_18']['AP50']:.6f}")
    print(f"Paper AP (Annex): {results['annex_18']['classes']['paper']['AP']:.6f}")

    print("\nPer-Class AP (Combined 65):")
    for name, stats in results["combined_65"]["classes"].items():
        ap_val = f"{stats['AP']:.6f}" if stats['AP'] is not None else "n/a"
        ar_val = f"{stats['AR100']:.6f}" if stats['AR100'] is not None else "n/a"
        print(f"  {name:<18} (GT: {stats['objects']:>2}): AP = {ap_val:<8} | AR100 = {ar_val}")

    # Save results
    save_name = f"eval_{args.checkpoint}_{args.shortest}_{args.longest}.json"
    save_path = args.run_dir / save_name
    atomic_json(save_path, results)
    print(f"\nDetailed evaluation saved to {save_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
