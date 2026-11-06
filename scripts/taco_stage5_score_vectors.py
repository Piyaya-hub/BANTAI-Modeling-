"""Bounded extraction of Stage 5 per-query nine-class scores for D-078.

This command performs exactly one validation forward per combined_65 image,
with no train split loading, backward pass, optimizer, or test split access.
All new output is placed in a unique private runs/training folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoProcessor, GroundingDinoForObjectDetection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.configuration import MODEL_ID, REVISION, strict_load
from scripts.persist_stage5_soup import verify_bundle
from scripts.train_taco_stage5 import freeze_backbones, load_verified_stage5_checkpoint
from scripts.training_smoke import SNAPSHOT, loading_ok, require_training_environment, state_hash
from scripts.training_adapter_v2 import EncoderOutputAdapterV2
from scripts.taco_training_labels_v2 import LABELS, validate_taxonomy
from scripts.taco_training_data_v2 import CocoTrainingDatasetV2, get_batch
from scripts.taco_evaluation_v3 import PROTOCOL, decode_predictions
from scripts.training_smoke import move_batch

RUN_DIR = ROOT / "runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14"
BUNDLE = RUN_DIR / "checkpoints/best-inference"
EXPECTED_HASH = "cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422"
SAVED = RUN_DIR / "soup_tier2_d074_champion_5050.json"
EXPORT = ROOT / "exports/taco/strict-clean-stage5-tier2-v1"
RIGHTS = ROOT / "work/stage5-paper/rights_tier2.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    require_training_environment()
    validate_taxonomy()
    if not torch.cuda.is_available():
        raise RuntimeError("D-078 authorized bounded extraction requires the verified CUDA runtime")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    manifest = verify_bundle(BUNDLE)
    if manifest["parameter_sha256"] != EXPECTED_HASH:
        raise ValueError("D-076 champion parameter SHA-256 mismatch")
    saved = strict_load(SAVED)
    source_checkpoint_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(BUNDLE.rglob("*")) if path.is_file()
    }
    protected_sources = [ROOT / "config/taco_taxonomy.v2.json"] + [
        ROOT / "scripts" / name for name in (
            "train_taco.py", "taco_training_data.py", "taco_evaluation.py",
            "training_checkpoint.py", "training_adapter.py", "training_smoke.py")]
    protected_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in protected_sources}
    dataset = CocoTrainingDatasetV2(EXPORT, "validation", RIGHTS)
    if len(dataset) != 65 or dataset.context["splits"]["validation"]["images"] != 65:
        raise ValueError("Expected the frozen combined_65 validation split")
    # Access only validation metadata and record the exact approved photo set.
    approved_photos = {(dataset.root / row["file_name"]).resolve() for row in dataset.images}
    if len(approved_photos) != 65:
        raise ValueError("Unexpected validation image paths")

    model, info = GroundingDinoForObjectDetection.from_pretrained(
        SNAPSHOT, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, disable_custom_kernels=True,
        output_loading_info=True,
    )
    loading_ok(info)
    processor = AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    freeze_backbones(model)
    proof = load_verified_stage5_checkpoint(model, BUNDLE)
    if proof["parameter_sha256"] != EXPECTED_HASH:
        raise ValueError("Loaded checkpoint parameter hash differs from D-076")
    model.to("cuda").eval()
    adapter = EncoderOutputAdapterV2(model, processor.tokenizer)

    calls = 0

    def count_forward(module, args, kwargs):
        nonlocal calls
        calls += 1
        if calls > 65:
            raise RuntimeError("D-078 forward budget exceeded")
        pixel = kwargs.get("pixel_values")
        if pixel is None or tuple(pixel.shape) != (1, 3, 576, 576) or pixel.dtype != torch.float32:
            raise ValueError("Expected one FP32 384/576 contract forward")

    handle = model.register_forward_pre_hook(count_forward, with_kwargs=True)
    vectors = np.empty((65, 900, 9), dtype=np.float32)
    ids, raw_all = [], []
    try:
        with torch.inference_mode():
            for index in range(65):
                batch, geometry = get_batch(dataset, [index], processor, 384, 576)
                batch = move_batch(batch, "cuda")
                result = adapter(model, batch)
                from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps
                spans = build_label_maps(result.outputs.logits, batch["input_ids"])[0]
                probabilities = result.outputs.logits[0].sigmoid()
                per_class = torch.stack([probabilities[:, mask.bool()].mean(-1) for mask in spans], dim=-1)
                if tuple(per_class.shape) != (900, 9):
                    raise ValueError("Expected 900 query by 9 class score matrix")
                image_id = int(geometry[0]["image_id"])
                ids.append(image_id)
                vectors[index] = per_class.detach().cpu().numpy().astype(np.float32, copy=False)
                predictions, raw = decode_predictions(result.outputs, batch, geometry, processor.tokenizer)
                raw_all.extend(raw)
                del result, batch, geometry, predictions, raw, per_class, probabilities
        torch.cuda.synchronize()
    finally:
        handle.remove()
    if calls != 65 or dataset.photo_reads != 65:
        raise ValueError(f"Expected exactly 65 forwards/photo reads; got {calls}/{dataset.photo_reads}")
    if source_checkpoint_hashes != {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(BUNDLE.rglob("*")) if path.is_file()
    } or protected_hashes != {str(path.relative_to(ROOT)): sha256(path) for path in protected_sources}:
        raise ValueError("Protected checkpoint/source hashes changed during extraction")

    saved_raw = saved["raw_queries"]
    if len(saved_raw) != 65 * 900 or len(raw_all) != len(saved_raw):
        raise ValueError("Raw-query count differs from the saved champion evaluation")
    saved_by_key = {(r["image_id"], r["query_id"]): r for r in saved_raw}
    max_score_error = 0.0
    for row in raw_all:
        expected = saved_by_key.get((row["image_id"], row["query_id"]))
        if expected is None or row["category_id"] != expected["category_id"]:
            raise ValueError("Top-1 class or query identity failed champion reproduction")
        error = abs(float(row["score"]) - float(expected["score"]))
        max_score_error = max(max_score_error, error)
        if error > 1e-7:
            raise ValueError("Top-1 winning score differs by more than 1e-7")

    out = ROOT / "runs/training" / ("stage5-score-vectors-" + uuid.uuid4().hex)
    out.mkdir(parents=False, exist_ok=False)
    vector_path = out / "query-scores.npz"
    np.savez_compressed(vector_path, image_ids=np.asarray(ids, dtype=np.int64),
                        scores=vectors, query_ids=np.arange(900, dtype=np.int16),
                        class_ids=np.arange(9, dtype=np.int8))
    result = {
        "schema_version": "bantai.stage5-query-score-vectors.v1",
        "authorization": "D-078",
        "state": "soup_tier2_d074_champion_5050",
        "diagnostic_only": True,
        "official_metrics_modified": False,
        "test_split_access": False,
        "training_or_backward_passes": 0,
        "device": "cuda",
        "parameter_sha256": proof["parameter_sha256"],
        "bundle_manifest_sha256": proof["manifest_sha256"],
        "saved_evaluation_sha256": sha256(SAVED),
        "validation_image_ids": ids,
        "validation_images": len(ids),
        "forwards": calls,
        "photo_reads": dataset.photo_reads,
        "classes": list(LABELS),
        "shape": list(vectors.shape),
        "score_definition": "mean ordinary-subword sigmoid probability for each class and each of 900 queries",
        "champion_top1_category_match": True,
        "champion_top1_max_abs_score_error": max_score_error,
        "artifact": vector_path.name,
        "artifact_sha256": sha256(vector_path),
        "validation_photo_sha256": {str(row["id"]): row["source_taco"]["file_sha256"] for row in dataset.images},
        "protocol": PROTOCOL,
        "protected_checkpoint_bundle_sha256": source_checkpoint_hashes,
        "protected_core_and_taxonomy_sha256": protected_hashes,
    }
    (out / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "shape", "forwards", "photo_reads", "parameter_sha256",
        "champion_top1_max_abs_score_error", "artifact", "artifact_sha256")}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
