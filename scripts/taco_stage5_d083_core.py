"""Bounded D-083 resolution, continuation, soup, and CPU diagnostic workflow."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

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
from scripts.taco_stage5_material_hierarchy import evaluate_case

EXPECTED_D076 = "cd29260abd5bd1675a2637ed365f4c1821dc1753cf37e852d4e3f1ab9f49a422"
EXPECTED_D074 = "fc0c497e1163509e96ec3a72c1f85729f18336aa1f688eee998c3c29031465eb"
D076_RUN = ROOT / "runs/training/stage5-tier2-6d4b34c076714947ae9bbfe8e3521c14"
D076_BUNDLE = D076_RUN / "checkpoints/best-inference"
D074_BUNDLE = ROOT / "runs/training/stage5-tier1-69d61fae160544468f155d34f0cb6337/checkpoints/soup-tier1-hr-blend2575"
D083_EXPORT = ROOT / "exports/taco/strict-clean-stage5-d083-core-v1"
D083_RIGHTS = ROOT / "work/stage5-paper/rights_d083_core.json"
D076_RIGHTS = ROOT / "work/stage5-paper/rights_tier2.json"
LEGACY_JSON = ROOT / "exports/taco/strict-clean-stage2-v1/validation.json"
PROTECTED_CORE_NAMES = ("train_taco.py", "taco_training_data.py", "taco_evaluation.py",
                        "training_checkpoint.py", "training_adapter.py", "training_smoke.py")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_snapshot(exclude_run=None):
    """Digest protected code, taxonomies, exports (except test.json), rights, bundles."""
    exclude_run = Path(exclude_run).resolve() if exclude_run else None
    rows = {}
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


def verify_snapshot(before, exclude_run):
    after = protected_snapshot(exclude_run)
    if before != after:
        changed = sorted(set(before) ^ set(after) | {key for key in before.keys() & after.keys()
                                                      if before[key] != after[key]})
        raise AssertionError("Protected artifact hash drift: " + ", ".join(changed[:12]))
    return {"files": len(before), "unchanged": True,
            "snapshot_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()}


def _bundle_manifest_hashes(bundle):
    bundle = Path(bundle)
    manifest_name = "soup_manifest.json" if (bundle / "soup_manifest.json").is_file() else "checkpoint_manifest.json"
    manifest = strict_load(bundle / manifest_name)
    for name, expected in manifest["files"].items():
        if _sha(bundle / name) != expected:
            raise ValueError(f"Checkpoint file changed: {bundle/name}")
    return {"manifest": str(bundle / manifest_name), "manifest_sha256": _sha(bundle / manifest_name),
            "parameter_sha256": manifest.get("parameter_sha256"), "files": manifest["files"]}


def _runtime_model(device="cuda"):
    import torch
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    require_training_environment()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not torch.cuda.is_available() or device != "cuda":
        raise RuntimeError("D-083 requires the explicitly authorized CUDA device; no CPU fallback")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce GTX 1660 Ti":
        raise RuntimeError("Unexpected CUDA device; stop before model execution")
    model, loading = GroundingDinoForObjectDetection.from_pretrained(
        SNAPSHOT, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        disable_custom_kernels=True, torch_dtype=torch.float32, output_loading_info=True)
    loading_ok(loading)
    processor = AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
    freeze_backbones(model)
    adapter = EncoderOutputAdapterV2(model, processor.tokenizer)
    return model, processor, adapter


def _install_access_guard(dataset_root, allowed_photos, run_dir):
    allowed = {Path(p).resolve() for p in allowed_photos}
    dataset_root = Path(dataset_root).resolve()
    run_dir = Path(run_dir).resolve()
    def guard(event, args):
        if event in ("os.remove", "os.rename", "os.rmdir", "os.mkdir"):
            paths = args[:2] if event == "os.rename" else args[:1]
            for value in paths:
                if isinstance(value, (str, bytes, os.PathLike)):
                    path = Path(os.fsdecode(value)).resolve()
                    if path.is_relative_to(ROOT) and not path.is_relative_to(run_dir):
                        raise AssertionError("D-083 may create/mutate only its new private run directory")
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name.lower() == "test.json":
            raise AssertionError("D-083 blocks test.json reads and writes")
        if path.is_relative_to(dataset_root) and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            if path not in allowed:
                raise AssertionError("D-083 attempted photo access outside authorized train/validation IDs")
        flags = args[2] if len(args) > 2 else 0
        writing = isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        if writing and path.is_relative_to(ROOT) and not path.is_relative_to(run_dir):
            raise AssertionError("D-083 process write outside its new private run directory")
    sys.addaudithook(guard)


def _class_score_vectors(outputs, input_ids):
    import torch
    from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps
    maps = build_label_maps(outputs.logits, input_ids)
    if any(len(mapping) != 9 for mapping in maps):
        raise ValueError("Nine prompt spans required for D-083 vectors")
    scores = outputs.logits.sigmoid()
    return torch.stack([torch.stack([scores[i, :, mask[:scores.shape[-1]].bool()].mean(-1)
                                     for mask in maps[i]], dim=-1)
                        for i in range(scores.shape[0])], dim=0)


def evaluate_d083(model, processor, adapter, dataset, device, shortest=384, longest=576,
                  *, forward_counter=None):
    """Run exactly 65 nine-class + 47 separate original-eight prompt forwards."""
    import torch
    from tqdm import tqdm
    from scripts.taco_evaluation_v3 import PROTOCOL as PROTOCOL9
    from scripts.training_smoke import checked_loss
    start = time.perf_counter()
    previous = model.training
    model.eval()
    predictions, raw_queries, losses, vectors = [], [], [], {}
    try:
        with torch.inference_mode():
            for index in tqdm(range(len(dataset)), desc=f"D-083 validation {shortest}/{longest}", unit="image"):
                batch, geometry = get_batch(dataset, [index], processor, shortest, longest)
                if tuple(batch["pixel_values"].shape) != (1, 3, longest, longest) or batch["pixel_values"].dtype != torch.float32:
                    raise ValueError("D-083 input geometry/precision contract changed")
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
            raise ValueError("D-083 validation does not contain exact legacy47 IDs")
        by_id = {row["id"]: i for i, row in enumerate(dataset.images)}
        class LegacyView:
            data = old_data
            images = old_data["images"]
            def __len__(self):
                return len(self.images)
            def __getitem__(self, index):
                ident = self.images[index]["id"]
                image, _ = dataset[by_id[ident]]
                return image, {"image_id": ident, "annotations": [a for a in self.data["annotations"]
                                                                      if a["image_id"] == ident]}
        legacy = evaluate_legacy(model, processor, EncoderOutputAdapter(model, processor.tokenizer),
                                 LegacyView(), device, shortest, longest)
        if forward_counter is not None and forward_counter() != 112:
            raise AssertionError(f"D-083 validation forward count is {forward_counter()}, expected 112")
        return {"combined_65": combined, "legacy_47": legacy["metrics"],
                "legacy_47_evaluation": legacy, "legacy_47_protocol": "bantai.taco-evaluation.v2; original eight-class prompt",
                "combined_65_protocol": PROTOCOL9, "mean_loss_per_image": sum(losses) / len(losses),
                "validation_forwards": 112, "raw_queries": raw_queries, "predictions": predictions,
                "class_vectors": vectors,
                "elapsed_seconds": time.perf_counter() - start}
    finally:
        model.train(previous)


def save_vectors(path, vectors):
    image_ids = sorted(vectors)
    array = np.stack([vectors[i] for i in image_ids]).astype(np.float32, copy=False)
    if array.shape != (65, 900, 9):
        raise ValueError(f"Expected (65,900,9) vectors, got {array.shape}")
    np.savez_compressed(path, image_ids=np.asarray(image_ids, dtype=np.int64), scores=array)
    return {"path": str(path), "shape": list(array.shape), "dtype": str(array.dtype), "sha256": _sha(path)}


def _metrics_snapshot(run_dir, name, result):
    vector_meta = save_vectors(run_dir / f"scores-{name}.npz", result["class_vectors"])
    value = {key: val for key, val in result.items() if key != "class_vectors"}
    value["score_vectors"] = vector_meta
    atomic_json(run_dir / f"{name}.json", value)
    return value


def _evaluate_materials(val_data, states_by_name):
    from scripts.taco_stage5_material_hierarchy import SCHEMES, MODES
    cases = []
    for state, rows in states_by_name.items():
        for scheme in ("bantai-5-category", "5-material", "4-material"):
            for mode in MODES:
                case = evaluate_case(val_data, rows, scheme, mode)
                cases.append({"state": state, "scheme": scheme, "nms": mode, **case})
    return {"diagnostic_only": True, "official_metrics_unchanged": True, "cases": cases}


def _consensus_audit(val_data, raw_rows, scores):
    from collections import defaultdict
    from scripts.taco_stage5_consensus_tiebreak import apply_consensus_tiebreak
    from scripts.taco_stage5_consensus_audit import official_ap, strict_counts
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
    return {"profile": "delta=.016; 3 plastic subtypes >= .20; top1_score",
            "changed_queries": len(changed), "query_changes": changed,
            "all_score_bbox_query_values_preserved": True,
            "baseline": {scheme: {"metrics": official_ap(val_data, baseline, scheme),
                                  "strict": {str(t): strict_counts(val_data, baseline, scheme, t)
                                             for t in thresholds}}
                         for scheme in schemes},
            "consensus": {scheme: {"metrics": official_ap(val_data, adjusted, scheme),
                                   "strict": {str(t): strict_counts(val_data, adjusted, scheme, t)
                                              for t in thresholds}}
                          for scheme in schemes}}

def _save_inference_bundle(model, processor, adapter, run_dir, state, parameter_sha256, eval_path, recipe):
    from scripts.training_smoke import digest
    folder = run_dir / "checkpoints/best-inference"
    folder.mkdir(parents=True, exist_ok=False)
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("D-083 inference serialization requires CPU model tensors")
    model.eval()
    model.save_pretrained(folder, safe_serialization=True)
    processor.save_pretrained(folder)
    atomic_json(folder / "training_adapter.json", adapter.manifest(model))
    files = {path.name: digest(path) for path in folder.iterdir() if path.is_file()}
    manifest = {"schema_version": "bantai.stage5-soup.v1", "status": "complete",
                "authorization": "D-083", "resume_supported": False,
                "training_resume_supported": False, "purpose": "private_inference_replay_not_promoted",
                "parameter_sha256": parameter_sha256, "frozen_parameter_sha256": state_hash(model, True),
                "best_state": state, "recipe": recipe, "evaluation_sha256": digest(eval_path),
                "trainable_tensors_averaged": 558 if "soup" in state else 0,
                "files": files}
    atomic_json(folder / "soup_manifest.json", manifest)
    for name, expected in manifest["files"].items():
        if digest(folder / name) != expected:
            raise ValueError("D-083 inference bundle file hash mismatch")
    return folder, manifest


def run_resolution(run_dir):
    import torch
    from scripts.train_taco_stage5 import load_verified_stage5_checkpoint
    before = protected_snapshot(exclude_run=run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(run_dir / "integrity-before.json", before)
    source_bundle = _bundle_manifest_hashes(D076_BUNDLE)
    if source_bundle["parameter_sha256"] != EXPECTED_D076:
        raise ValueError("D-076 champion parameter hash differs")
    stage2_val = json.loads(LEGACY_JSON.read_text(encoding="utf-8"))
    dataset = CocoTrainingDatasetV2(D076_RUN / ".." / ".." / ".." / "exports/taco/strict-clean-stage5-tier2-v1",
                                    "validation", D076_RIGHTS)
    # Resolve the explicit source export, avoiding any split-discovery ambiguity.
    dataset = CocoTrainingDatasetV2(ROOT / "exports/taco/strict-clean-stage5-tier2-v1",
                                    "validation", D076_RIGHTS)
    if len(dataset) != 65 or len(dataset.data["annotations"]) != 109 or len(stage2_val["images"]) != 47:
        raise ValueError("D-076 validation boundary changed")
    allowed = {dataset.root / row["file_name"] for row in dataset.images}
    _install_access_guard(dataset.root, allowed, run_dir)
    model, processor, adapter = _runtime_model("cuda")
    proof = load_verified_stage5_checkpoint(model, D076_BUNDLE)
    if proof["parameter_sha256"] != EXPECTED_D076:
        raise ValueError("Loaded D-076 champion parameter hash mismatch")
    model.to("cuda")
    calls = {"total": 0}
    def prehook(module, args, kwargs):
        if calls["total"] >= 112:
            raise AssertionError("448/672 evaluation forward budget exceeded")
        pv = kwargs["pixel_values"]
        if tuple(pv.shape) != (1, 3, 672, 672) or pv.dtype != torch.float32:
            raise ValueError("D-083 448/672 FP32 geometry changed")
        calls["total"] += 1
    handle = model.register_forward_pre_hook(prehook, with_kwargs=True)
    try:
        result = evaluate_stage5_448(model, processor, adapter, dataset, device="cuda")
    finally:
        handle.remove()
    if calls["total"] != 112 or result["validation_forwards"] != 112:
        raise AssertionError("Part A must complete exactly 112 validation forwards")
    model.to("cpu")
    torch.cuda.empty_cache()
    val_data = dataset.data
    reference = strict_load(D076_RUN / "soup_tier2_d074_champion_5050.json")
    ref_rows = reference["raw_queries"]
    ref_material = evaluate_case(val_data, ref_rows, "bantai-5-category", "raw")
    hi_material = evaluate_case(val_data, result["raw_queries"], "bantai-5-category", "raw")
    comparison = {"state": "D076 champion", "resolution": "448/672", "forwards": calls["total"],
                  "checkpoint_parameter_sha256": proof["parameter_sha256"],
                  "reference_resolution": "384/576", "reference_state": "D076 champion",
                  "official_9_specific_combined_65": {"384/576": reference["combined_65"],
                                                       "448/672": result["combined_65"]},
                  "official_8_class_legacy_47": {"384/576": reference["legacy_47"],
                                                 "448/672": result["legacy_47"]},
                  "bantai_5_category_combined_65_raw": {"384/576": ref_material["metrics"],
                                                        "448/672": hi_material["metrics"]},
                  "bantai_5_coverage": {"384/576": ref_material["covered"],
                                        "448/672": hi_material["covered"]},
                  "source_checkpoint": source_bundle}
    atomic_json(run_dir / "resolution-448-672.json", result)
    atomic_json(run_dir / "resolution-comparison.json", comparison)
    integrity = verify_snapshot(before, run_dir)
    atomic_json(run_dir / "integrity-after-resolution.json", integrity)
    atomic_json(run_dir / "run-state.json", {"phase": "part-a-complete", "run_dir": str(run_dir),
                                               "resolution_forwards": calls["total"],
                                               "protected_artifacts_unchanged": True})
    model.to("cpu")
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    print(json.dumps({"run_dir": str(run_dir), "forwards": calls["total"],
                      "combined_AP_448_672": result["combined_65"]["AP"],
                      "legacy_AP_448_672": result["legacy_47"]["AP"],
                      "BANTAI5_AP_448_672": hi_material["metrics"]["AP"],
                      "integrity": integrity}, indent=2))
    return run_dir


def evaluate_stage5_448(model, processor, adapter, dataset, device):
    """Stage 5 official evaluator pinned to the D-083 448/672 geometry."""
    from scripts.taco_evaluation_v3 import evaluate_stage5
    return evaluate_stage5(model, processor, adapter, dataset, device, shortest=448, longest=672)


def _save_eval(run_dir, state, result):
    vectors = result.pop("class_vectors")
    meta = save_vectors(run_dir / f"scores-{state}.npz", vectors)
    result["score_vectors"] = meta
    atomic_json(run_dir / f"{state}.json", result)
    return vectors, result


def train_d083(run_dir):
    import torch
    from scripts.train_taco_stage5 import load_verified_stage5_checkpoint
    run_dir = Path(run_dir).resolve()
    before = strict_load(run_dir / "integrity-before.json")
    verify_snapshot(before, run_dir)
    output = run_dir / "checkpoints"
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError("D-083 checkpoint directory is nonempty; refusing a second training run")
    else:
        output.mkdir()
    d073 = strict_load(ROOT / "runs/training/stage5-data-expansion-audit-5412a13c0258467ba33a1f475cc4b65c/summary.json")
    export_manifest = strict_load(D083_EXPORT / "preparation_manifest.json")
    train_meta = strict_load(D083_EXPORT / "train.json")
    val_meta = strict_load(D083_EXPORT / "validation.json")
    if len(train_meta["images"]) != 379 or len(train_meta["annotations"]) != 731:
        raise ValueError("D-083 training export counts differ from its authorization")
    if len(val_meta["images"]) != 65 or len(val_meta["annotations"]) != 109:
        raise ValueError("D-083 validation export changed")
    d073_frozen = d073["frozen"]
    if set(i["id"] for i in val_meta["images"]) != set(d073_frozen["validation"]["ids"]):
        raise ValueError("D-083 validation IDs differ from the frozen 65-image set")
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
    train_ds = CocoTrainingDatasetV2(D083_EXPORT, "train", D083_RIGHTS)
    val_ds = CocoTrainingDatasetV2(D083_EXPORT, "validation", D083_RIGHTS)
    if (len(train_ds), len(train_ds.data["annotations"])) != (379, 731):
        raise ValueError("D-083 loader dataset counts changed")
    if (len(val_ds), len(val_ds.data["annotations"])) != (65, 109):
        raise ValueError("D-083 loader validation counts changed")
    d076_proof = _bundle_manifest_hashes(D076_BUNDLE)
    d074_proof = _bundle_manifest_hashes(D074_BUNDLE)
    if d076_proof["parameter_sha256"] != EXPECTED_D076 or d074_proof["parameter_sha256"] != EXPECTED_D074:
        raise ValueError("D083 source champion parameter digest mismatch")
    model, processor, adapter = _runtime_model("cuda")
    warmstart = load_verified_stage5_checkpoint(model, D076_BUNDLE)
    if warmstart["parameter_sha256"] != EXPECTED_D076:
        raise ValueError("Warm-start did not load D076 champion exactly")
    model.to("cuda")
    frozen_hash = state_hash(model, True)
    call_counts = {"training": 0, "validation": 0}
    def count_pre_hook(module, args, kwargs):
        mode = "training" if module.training and torch.is_grad_enabled() else "validation"
        limit = 379 if mode == "training" else 336
        if call_counts[mode] >= limit:
            raise AssertionError(f"D-083 {mode} forward budget exceeded")
        pixel_values = kwargs["pixel_values"]
        if tuple(pixel_values.shape) != (1, 3, 576, 576) or pixel_values.dtype != torch.float32:
            raise ValueError("D-083 384/576 FP32 forward geometry changed")
        call_counts[mode] += 1
    handle = model.register_forward_pre_hook(count_pre_hook, with_kwargs=True)
    implementation = {name: sha256(ROOT / name) for name in SOURCES}
    stage5_files = ("training_adapter_v2.py", "taco_training_data_v2.py", "taco_training_labels_v2.py",
                    "taco_evaluation_v3.py", "train_taco_stage5.py", "taco_stage5_d083_core.py")
    contract = {"authorization": "D-083", "run_id": run_dir.name, "labels": list(LABELS), "prompt": PROMPT,
                "training_images": 379, "training_annotations": 731, "validation_images": 65,
                "validation_annotations": 109, "shortest": 384, "longest": 576, "square_pad": 576,
                "epochs": 1, "batch_size": 1, "seed": 42,
                "optimizer": {"name": "AdamW", "lr": 1.5e-6, "weight_decay": 1e-4, "foreach": False},
                "clip_norm": 1.0, "precision": "FP32", "TF32": False, "device": torch.cuda.get_device_name(0),
                "warmstart": warmstart, "frozen_parameter_sha256": frozen_hash,
                "rights_sha256": train_ds.rights["record_sha256"], "dataset_context": train_ds.context,
                "checkpoint_every_steps": 100, "test_access": False,
                "implementation_sha256": implementation,
                "stage5_implementation_sha256": {name: sha256(ROOT / "scripts" / name) for name in stage5_files}}
    atomic_json(run_dir / "contract.json", contract, replace=True)
    atomic_json(run_dir / "training-run-state.json", {"phase": "training", "status": "running", "contract": contract}, replace=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=1.5e-6, weight_decay=1e-4, foreach=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    steps = []
    batches = epoch_batches(379, 1, 42, 0)
    if len(batches) != 379 or any(len(batch) != 1 for batch in batches):
        raise ValueError("Unexpected D-083 deterministic sampler length")
    model.train()
    for step, batch_indexes in enumerate(batches, start=1):
        index = batch_indexes[0]
        batch, geometry = get_batch(train_ds, [index], processor, 384, 576)
        if geometry[0]["image_id"] != train_ds.images[index]["id"]:
            raise ValueError("D-083 sampler-to-image identity mismatch")
        batch = move_batch(batch, "cuda")
        row = train_step(model, adapter, optimizer, batch)
        row.update(step=step, batch_index=index, image_id=train_ds.images[index]["id"])
        steps.append(row)
        with (run_dir / "steps.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        del batch
        if step % 50 == 0 or step == 379:
            print(f"D-083 update {step}/379 loss={row['loss']:.5f} peak={torch.cuda.max_memory_allocated()/2**20:.0f} MiB",
                  flush=True)
        if step % 100 == 0:
            if state_hash(model, True) != frozen_hash:
                raise ValueError("Frozen parameters changed before checkpoint")
            position = {"epoch": 0, "next_batch": step, "global_step": step}
            save_checkpoint(output / f"step-{step:06d}", model, processor, adapter, optimizer,
                            contract, position, {"frozen_parameter_sha256": frozen_hash})
            verify_checkpoint(output / f"step-{step:06d}", contract)
    if call_counts["training"] != 379 or state_hash(model, True) != frozen_hash:
        raise AssertionError("D-083 training forward count/frozen hash mismatch")
    epoch_position = {"epoch": 1, "next_batch": 0, "global_step": 379}
    save_checkpoint(output / "epoch-01", model, processor, adapter, optimizer,
                    contract, epoch_position, {"frozen_parameter_sha256": frozen_hash})
    verify_checkpoint(output / "epoch-01", contract)
    e1_hash = state_hash(model)
    e1_state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
                if parameter.requires_grad}
    model.eval()
    validation_snapshot = rng_state()
    try:
        result_e1 = evaluate_d083(model, processor, adapter, val_ds, "cuda", 384, 576,
                                  forward_counter=lambda: call_counts["validation"])
    finally:
        restore_rng(validation_snapshot)
    if call_counts["validation"] != 112:
        raise AssertionError("D-083 E1 must consume 112 validation forwards")
    result_e1["parameter_sha256"] = e1_hash
    e1_vectors, saved_e1 = _save_eval(run_dir, "validation-d083-core-e1", result_e1)
    evaluations = {"D083-Core E1": saved_e1}
    best = {"state": "D083-Core E1", "AP": saved_e1["combined_65"]["AP"],
            "parameter_sha256": e1_hash, "vectors": e1_vectors,
            "evaluation": saved_e1, "recipe": {"D083-Core E1": 1.0}}
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    del optimizer
    model.to("cpu")
    import gc
    gc.collect(); torch.cuda.empty_cache()

    ref_states = {}
    for name, bundle, expected in (("D074", D074_BUNDLE, EXPECTED_D074),
                                   ("D076", D076_BUNDLE, EXPECTED_D076)):
        proof = load_verified_stage5_checkpoint(model, bundle)
        if proof["parameter_sha256"] != expected:
            raise ValueError(f"{name} source bundle did not load with its bound parameter hash")
        ref_states[name] = {key: parameter.detach().cpu().clone()
                            for key, parameter in model.named_parameters() if parameter.requires_grad}
    recipes = (("soup_d083_d074_5050", "D074", EXPECTED_D074),
               ("soup_d083_d076_5050", "D076", EXPECTED_D076))
    soup_states = {}
    for label, reference, reference_hash in recipes:
        prior_validation = call_counts["validation"]
        averaged = average_trainable({"reference": ref_states[reference], "d083": e1_state},
                                     {"reference": .5, "d083": .5})
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(averaged[name])
        parameter_hash = state_hash(model)
        if state_hash(model, True) != frozen_hash:
            raise ValueError("D-083 soup modified frozen parameters")
        model.to("cuda")
        validation_snapshot = rng_state()
        try:
            result = evaluate_d083(model, processor, adapter, val_ds, "cuda", 384, 576,
                                   forward_counter=lambda: call_counts["validation"] - prior_validation)
        finally:
            restore_rng(validation_snapshot)
        result["parameter_sha256"] = parameter_hash
        vectors, saved = _save_eval(run_dir, label, result)
        evaluations[label] = saved
        soup_states[label] = averaged
        if saved["combined_65"]["AP"] > best["AP"]:
            best = {"state": label, "AP": saved["combined_65"]["AP"],
                    "parameter_sha256": parameter_hash, "vectors": vectors,
                    "evaluation": saved,
                    "recipe": {reference: .5, "D083-Core E1": .5},
                    "reference_parameter_sha256": reference_hash}
        model.to("cpu")
        gc.collect(); torch.cuda.empty_cache()
    handle.remove()
    if call_counts != {"training": 379, "validation": 336}:
        raise AssertionError(f"D-083 forward contract mismatch: {call_counts}")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise AssertionError("TF32 was unexpectedly enabled")

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(best["vectors_trainable"][name] if "vectors_trainable" in best
                                else (e1_state[name] if best["state"] == "D083-Core E1" else
                                      soup_states[best["state"]][name]))
    # Ensure the model tensors serialized exactly match the measured winning state.
    if state_hash(model) != best["parameter_sha256"] or state_hash(model, True) != frozen_hash:
        raise ValueError("Best state restoration failed before inference bundle serialization")
    best_eval_path = run_dir / ("validation-d083-core-e1.json" if best["state"] == "D083-Core E1"
                                else best["state"] + ".json")
    bundle, bundle_manifest = _save_inference_bundle(
        model, processor, adapter, run_dir, best["state"], best["parameter_sha256"], best_eval_path, best["recipe"])
    best_vectors_meta = save_vectors(run_dir / "best-score-vectors.npz", best["vectors"])

    val_data = val_ds.data
    rows_by_state = {name: saved["raw_queries"] for name, saved in evaluations.items()}
    material_audit = _evaluate_materials(val_data, rows_by_state)
    atomic_json(run_dir / "material-hierarchy-audit.json", material_audit)
    consensus = _consensus_audit(val_data, best["evaluation"]["raw_queries"], best["vectors"])
    atomic_json(run_dir / "d079-score-preserving-consensus.json", consensus)
    peak_vram = torch.cuda.max_memory_allocated()
    report = {"authorization": "D-083", "status": "complete", "run_id": run_dir.name,
              "contract": contract, "training_updates": 379, "training_forwards": call_counts["training"],
              "validation_forwards": call_counts["validation"], "training_photo_reads": train_ds.photo_reads,
              "validation_photo_reads": val_ds.photo_reads, "optimizer": contract["optimizer"],
              "training_mean_loss": sum(row["loss"] for row in steps) / len(steps),
              "elapsed_seconds": time.perf_counter() - started, "peak_VRAM_bytes": peak_vram,
              "warmstart_parameter_sha256": warmstart["parameter_sha256"],
              "epoch_parameter_sha256": e1_hash, "states": {
                  name: {"parameter_sha256": value["parameter_sha256"],
                         "combined_65": value["combined_65"], "legacy_47": value["legacy_47"],
                         "mean_loss_per_image": value["mean_loss_per_image"],
                         "validation_forwards": value["validation_forwards"],
                         "score_vectors": value["score_vectors"]}
                  for name, value in evaluations.items()},
              "best_state": {key: value for key, value in best.items() if key not in ("vectors", "evaluation")},
              "best_bundle": str(bundle), "best_bundle_parameter_sha256": bundle_manifest["parameter_sha256"],
              "best_score_vectors": best_vectors_meta,
              "material_hierarchy_audit": "material-hierarchy-audit.json",
              "d079_consensus_audit": "d079-score-preserving-consensus.json",
              "source_bundles": {"D074": d074_proof, "D076": d076_proof},
              "frozen_parameters_unchanged": state_hash(model, True) == frozen_hash,
              "test_json_opened": False, "heldout_test_photos_opened": 0,
              "core_sources_unchanged": {name: sha256(ROOT / name) == expected
                                          for name, expected in contract["implementation_sha256"].items()}}
    atomic_json(run_dir / "report.json", report)
    after = verify_snapshot(before, run_dir)
    atomic_json(run_dir / "integrity-after-training.json", after)
    atomic_json(run_dir / "training-run-state.json", {"phase": "training-and-evaluation-complete",
                                                        "forward_counts": call_counts,
                                                        "protected_artifacts_unchanged": True}, replace=True)
    model.to("cpu"); gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({"run_dir": str(run_dir), "training_updates": 379,
                      "validation_forwards": call_counts["validation"], "best_state": report["best_state"],
                      "peak_vram_mib": peak_vram / 2**20, "material_audit_cases": len(material_audit["cases"]),
                      "consensus_reassignments": consensus["changed_queries"], "integrity": after}, indent=2))


def cli():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    resolution = sub.add_parser("resolution", help="D-083 Part A: 448/672 current champion evaluation")
    resolution.add_argument("--run-dir", type=Path, default=None)
    train = sub.add_parser("train", help="D-083 Part B: one epoch plus two CPU soups")
    train.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "resolution":
        run_dir = args.run_dir.resolve() if args.run_dir else ROOT / "runs/training" / ("stage5-d083-core-" + uuid.uuid4().hex)
        run_resolution(run_dir)
    else:
        train_d083(args.run_dir)


if __name__ == "__main__":
    cli()
