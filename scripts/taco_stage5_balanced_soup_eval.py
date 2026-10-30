"""D077: CPU-averaged balanced Stage5 soups, bounded validation-only evaluation.

The script performs no training/update. It evaluates exactly two averaged states
on the frozen 65-image combined nine-class validation view (65 forwards each).
It never opens test.json or test-split photographs.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from unittest.mock import patch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from scripts.taco_training_data_v2 import CocoTrainingDatasetV2, get_batch
from scripts.taco_training_labels_v2 import LABELS, PROMPT
from scripts.training_adapter_v2 import EncoderOutputAdapterV2
from scripts.training_smoke import SNAPSHOT, digest, state_hash, loading_ok, move_batch, checked_loss
from scripts.taco_weight_integration import average_trainable
from scripts.taco_evaluation_v3 import PROTOCOL, coco_metrics, decode_predictions
from scripts import train_taco_stage5 as stage5

RUN1 = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4"
TIER1 = ROOT / "runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337"
TIER2 = ROOT / "runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14"
EXPORT = ROOT / "exports/taco/strict-clean-stage5-tier2-v1"
RIGHTS = ROOT / "work/stage5-paper/rights_tier2.json"
CHECKPOINTS = {
    "run1_e1": RUN1 / "checkpoints/epoch-01",
    "run1_e2": RUN1 / "checkpoints/epoch-02",
    "tier1_e1": TIER1 / "checkpoints/epoch-01",
    "tier2_e1": TIER2 / "checkpoints/epoch-01",
}
EXPECTED_BLEND2575 = "5a8b832754cfbf16fa56e4742299c61a8f60cdba6cba9d6c9758ec85ad7b3e98"
PRIOR_FAILED_FORWARD_ATTEMPTS = 1


def evaluate_combined65(model, processor, adapter, dataset, device):
    """Run the official nine-specific combined validation view once per image."""
    predictions, raw_queries, losses = [], [], []
    model.eval()
    for index in range(len(dataset)):
        batch, geometry = get_batch(dataset, [index], processor, 384, 576)
        batch = move_batch(batch, device)
        result = adapter(model, batch)
        checked_loss(result, model.config)
        losses.append(float(result.loss))
        decoded, raw = decode_predictions(result.outputs, batch, geometry, processor.tokenizer)
        predictions.extend(decoded)
        raw_queries.extend(raw)
    metrics = coco_metrics(dataset.data, predictions,
                           img_ids=[row["id"] for row in dataset.images],
                           cat_ids=list(range(9)), label_names=LABELS)
    return {"protocol": PROTOCOL, "combined_65": metrics,
            "validation_forwards": len(dataset),
            "mean_loss_per_image": sum(losses) / len(losses),
            "predictions": predictions, "raw_queries": raw_queries}


def checkpoint_hashes():
    values = {}
    for name, folder in CHECKPOINTS.items():
        values[name] = {str(p.relative_to(ROOT)): digest(p)
                        for p in sorted(folder.rglob("*")) if p.is_file()}
    return values


def copy_trainable(model):
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()
            if name in names}


def main():
    if len(LABELS) != 9 or PROMPT.count(".") < 1:
        raise ValueError("Unexpected Stage5 label/prompt contract")
    if not EXPORT.is_dir() or not RIGHTS.is_file():
        raise FileNotFoundError("Required private Stage5 validation export/rights unavailable")
    for path in CHECKPOINTS.values():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing required checkpoint: {path}")

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = ROOT / "runs/training" / ("stage5-balanced-soup-eval-" + uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    before_bundles = checkpoint_hashes()
    protected = [ROOT / n for n in (
        "scripts/train_taco.py", "scripts/taco_training_data.py", "scripts/taco_evaluation.py",
        "scripts/training_checkpoint.py", "scripts/training_adapter.py", "scripts/training_smoke.py",
        "config/taco_taxonomy.v1.json", "config/taco_taxonomy.v2.json")]
    protected_before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    rights_before = digest(RIGHTS)
    export_before = {str(p.relative_to(ROOT)): digest(p) for p in (
        EXPORT / "preparation_manifest.json", EXPORT / "train.json", EXPORT / "validation.json")}

    val_ds = CocoTrainingDatasetV2(EXPORT, "validation", RIGHTS)
    if len(val_ds) != 65:
        raise ValueError("Expected the frozen 65-image validation set")
    val_photos = {(val_ds.root / row["file_name"]).resolve() for row in val_ds.images}
    output_abs = output.resolve()

    def file_guard(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.name.lower() == "test.json":
                raise AssertionError("D077 prohibits test.json access")
            if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                if path.is_relative_to(val_ds.root) and path not in val_photos:
                    raise AssertionError("Only frozen validation photos may be opened")
            flags = args[2] if len(args) > 2 else 0
            if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                if path.is_relative_to(ROOT) and not path.is_relative_to(output_abs):
                    raise AssertionError("D077 writes are restricted to its new output folder")
        elif event in ("os.remove", "os.rmdir", "os.rename", "os.mkdir"):
            paths = args[:2] if event == "os.rename" else args[:1]
            for value in paths:
                if isinstance(value, (str, bytes, os.PathLike)):
                    path = Path(os.fsdecode(value)).resolve()
                    if path.is_relative_to(ROOT) and not path.is_relative_to(output_abs):
                        raise AssertionError("D077 filesystem mutation escaped its output folder")

    sys.addaudithook(file_guard)
    model, info = GroundingDinoForObjectDetection.from_pretrained(
        SNAPSHOT, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, disable_custom_kernels=True, output_loading_info=True,
    )
    loading_ok(info)
    processor = AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True,
                                               trust_remote_code=False)
    stage5.freeze_backbones(model)
    base_frozen_hash = state_hash(model, True)
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    if len(trainable_names) != 558:
        raise ValueError("Expected the verified Stage5 558-parameter trainable set")

    proofs = {}
    source_states = {}
    for key in ("run1_e1", "run1_e2", "tier1_e1", "tier2_e1"):
        proofs[key] = stage5.load_verified_stage5_checkpoint(model, CHECKPOINTS[key])
        source_states[key] = copy_trainable(model)
        if state_hash(model, True) != base_frozen_hash:
            raise ValueError("A source checkpoint changed frozen parameters")
    blend = average_trainable({"e1": source_states["run1_e1"],
                               "e2": source_states["run1_e2"]},
                              {"e1": .25, "e2": .75})
    with torch.no_grad():
        named = dict(model.named_parameters())
        for name, value in blend.items():
            named[name].copy_(value)
    blend_hash = state_hash(model)
    if blend_hash != EXPECTED_BLEND2575:
        raise ValueError("Run1 blend2575 failed the recorded exact parameter hash")

    adapter = EncoderOutputAdapterV2(model, processor.tokenizer)
    calls = {"forward": 0}

    def count_forward(module, args, kwargs):
        calls["forward"] += 1
        if calls["forward"] + PRIOR_FAILED_FORWARD_ATTEMPTS > 224:
            raise AssertionError("D077 validation forward cap (224) exceeded")
        if kwargs["pixel_values"].dtype != torch.float32 or tuple(kwargs["pixel_values"].shape) != (1, 3, 576, 576):
            raise AssertionError("Expected one FP32 576x576 validation canvas")

    model.register_forward_pre_hook(count_forward, with_kwargs=True)
    recipes = (("balanced40_60_tier1", "tier1_e1"),
               ("balanced40_60_tier2", "tier2_e1"))
    results = {}
    frozen_hash = state_hash(model, True)
    for label, source in recipes:
        averaged = average_trainable({"blend2575": blend, "tier": source_states[source]},
                                     {"blend2575": .40, "tier": .60})
        with torch.no_grad():
            for name, value in averaged.items():
                named[name].copy_(value)
        parameter_hash = state_hash(model)
        if state_hash(model, True) != frozen_hash or frozen_hash != base_frozen_hash:
            raise ValueError("Balanced soup modified frozen parameters")
        model.to(device)
        model.eval()
        before = calls["forward"]
        with patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward prohibited")), \
             patch.object(torch.autograd, "backward", side_effect=AssertionError("backward prohibited")), \
             patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer step prohibited")), \
             torch.inference_mode():
            evaluation = evaluate_combined65(model, processor, adapter, val_ds, device)
        forwards = calls["forward"] - before
        if forwards != 65 or evaluation["validation_forwards"] != 65:
            raise ValueError(f"Each balanced state must use exactly 65 forwards, got {forwards}")
        if state_hash(model) != parameter_hash or state_hash(model, True) != frozen_hash:
            raise ValueError("Evaluation changed model weights")
        results[label] = {
            "recipe": {"blend2575": .40, source: .60},
            "parameter_sha256": parameter_hash,
            "forward_count": forwards,
            "combined_65": evaluation["combined_65"],
            "mean_loss_per_image": evaluation["mean_loss_per_image"],
        }
        evaluation["diagnostic_only"] = True
        evaluation["taxon_scope_note"] = "official 9-specific combined_65 view; no NMS"
        evaluation["weight_recipe"] = results[label]["recipe"]
        evaluation["parameter_sha256"] = parameter_hash
        (output / (label + ".json")).write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
        model.to("cpu")
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if calls["forward"] != 130 or calls["forward"] + PRIOR_FAILED_FORWARD_ATTEMPTS > 224:
        raise ValueError("Expected two combined_65 evaluations under the cumulative 224-attempt cap")
    if checkpoint_hashes() != before_bundles:
        raise ValueError("One or more existing source checkpoint bundles changed")
    if protected_before != {str(p.relative_to(ROOT)): digest(p) for p in protected}:
        raise ValueError("Protected sources/taxonomies changed")
    if rights_before != digest(RIGHTS) or export_before != {str(p.relative_to(ROOT)): digest(p) for p in (
            EXPORT / "preparation_manifest.json", EXPORT / "train.json", EXPORT / "validation.json")}:
        raise ValueError("Rights or export source changed")
    summary = {
        "schema_version": "bantai.stage5-balanced-soup-audit.v1",
        "diagnostic_only": True, "training_updates": 0,
        "optimizer_steps": 0, "backward_passes": 0,
        "test_json_opened": False, "test_photo_access": False,
        "validation_images": 65, "evaluation_scope": "combined_65 only; 65 forwards per state",
        "failed_forward_attempts_before_this_run": PRIOR_FAILED_FORWARD_ATTEMPTS,
        "validation_forwards_completed": calls["forward"],
        "validation_forward_attempts_cumulative": calls["forward"] + PRIOR_FAILED_FORWARD_ATTEMPTS,
        "forward_cap": 224,
        "device": device, "tf32": False, "precision": "FP32",
        "canvas": {"shortest": 384, "longest": 576, "square_pad": 576},
        "blend2575_parameter_sha256": blend_hash,
        "source_checkpoint_proofs": proofs,
        "base_frozen_parameter_sha256": base_frozen_hash,
        "source_checkpoint_bundles_unchanged": True,
        "protected_files_unchanged": True,
        "rights_and_export_unchanged": True,
        "results": results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "device": device,
                      "forwards": calls["forward"], "results": {
        key: value["combined_65"]["AP"] for key, value in results.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
