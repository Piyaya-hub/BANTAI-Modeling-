"""Stage 5 D-110: Video Frame Extraction, Standalone Input Image Processing & Three-Model Waste Detection.

Extracts frames from data/input/Motion_Extract_Testing.mp4 and performs
inference using Three-Model Stage 5 consensus:
  1. Warm-Start Baseline (Stage 4 blend at 384/576)
  2. Stage 5 Run 1 (256/384)
  3. D-095 Champion (soup 50/50 at 384/576)

Applies D-105 scene-level consensus and scale-normalized spatial grouping.
Saves hash-bound replay artifacts under runs/video_detection/motion_extract_testing/.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from scripts.taco_training_labels_v2 import LABELS, PROMPT
from scripts.taco_evaluation_v3 import decode_predictions
from scripts.taco_stage5_d095_box_contrastive import _class_score_vectors
from scripts.train_taco_stage5 import freeze_backbones, load_warmstart_blend
from scripts.taco_stage5_consensus_tiebreak import apply_scene_consensus
from scripts.taco_stage5_display_profiles import visible_rows
from scripts.taco_stage5_spatial_grouping import compute_spatial_grouping

CANONICAL_CATEGORIES = {
    0: "Potentially Hazardous",
    1: "Plastic",
    2: "Plastic",
    3: "Plastic",
    4: "Plastic",
    5: "Metal",
    6: "Plastic",
    7: "Plastic",
    8: "Paper",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    sample_fps: float = 1.0,
    max_frames: int = 30,
    start_id: int = 10001,
) -> tuple[dict, list[dict]]:
    """Extract still RGB JPEG frames from a video file using OpenCV."""
    import cv2

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_sha = _sha256(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    video_meta = {
        "path": str(video_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": video_sha,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 3),
        "file_size_bytes": os.path.getsize(video_path),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_step = max(1, int(round(fps / sample_fps))) if sample_fps > 0 else 1
    extracted: list[dict] = []
    current_frame = 0
    img_id = start_id

    while current_frame < total_frames and len(extracted) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame_bgr = cap.read()
        if not ret:
            break

        timestamp_sec = round(current_frame / fps, 3) if fps > 0 else 0.0
        file_name = f"frame_{len(extracted)+1:04d}_t{timestamp_sec:04.1f}s.jpg"
        file_path = output_dir / file_name

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        pil_img.save(file_path, quality=95)

        frame_sha = _sha256(file_path)
        extracted.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "frame_index": current_frame,
            "timestamp_sec": timestamp_sec,
            "file_sha256": frame_sha,
        })

        img_id += 1
        current_frame += frame_step

    cap.release()
    return video_meta, extracted


def run_single_model_inference(
    model,
    processor,
    images: list[tuple[int, Image.Image]],
    shortest: int,
    longest: int,
    device: str,
    desc: str = "Inference",
) -> tuple[dict[int, list[dict]], dict[int, list[dict]], dict[int, np.ndarray]]:
    """Run inference for a list of (image_id, pil_image) tuples."""
    import torch

    predictions_by_img: dict[int, list[dict]] = defaultdict(list)
    raw_queries_by_img: dict[int, list[dict]] = defaultdict(list)
    score_vectors_by_img: dict[int, np.ndarray] = {}

    model.eval()
    with torch.inference_mode():
        for img_id, pil_img in images:
            batch = processor(
                images=[pil_img],
                text=[PROMPT],
                padding=True,
                return_tensors="pt",
                size={"shortest_edge": shortest, "longest_edge": longest},
                pad_size={"height": longest, "width": longest},
            )
            geometry = [{"image_id": img_id, "original_wh": list(pil_img.size)}]
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

            outputs = model(**batch)
            preds, raw = decode_predictions(outputs, batch, geometry, processor.tokenizer)
            scores = _class_score_vectors(outputs, batch["input_ids"])[0].detach().cpu().numpy()

            predictions_by_img[img_id] = preds
            raw_queries_by_img[img_id] = raw
            score_vectors_by_img[img_id] = scores.astype(np.float32, copy=False)

            del batch, outputs

    return predictions_by_img, raw_queries_by_img, score_vectors_by_img


def execute_three_model_detection(
    frame_entries: list[dict],
    frames_dir: Path,
    device: str = "cuda",
) -> dict:
    """Run Warm-Start Baseline, Stage 5 Run 1, and D-095 Champion on extracted frames."""
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Loading frames from {frames_dir} on device {device}...", flush=True)
    loaded_images: list[tuple[int, Image.Image]] = []
    for entry in frame_entries:
        p = frames_dir / entry["file_name"]
        with Image.open(p) as img:
            loaded_images.append((entry["id"], img.convert("RGB")))

    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")

    # 1. Warm-Start Baseline (384/576)
    print("\n[1/3] Running Warm-Start Baseline (Stage 4 blend, 384/576)...", flush=True)
    t0 = time.perf_counter()
    baseline_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
    freeze_backbones(baseline_model)
    stage4_run = ROOT / "runs/training/taco-896e2f44146d4d54aee518e6087908ca"
    load_warmstart_blend(baseline_model, stage4_run)
    baseline_model.to(device)

    _, base_raw, _ = run_single_model_inference(
        baseline_model, processor, loaded_images,
        shortest=384, longest=576, device=device, desc="Baseline"
    )
    del baseline_model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print(f"  Warm-Start Baseline finished in {time.perf_counter() - t0:.2f}s", flush=True)

    # 2. Stage 5 Run 1 (256/384)
    print("\n[2/3] Running Stage 5 Run 1 (256/384)...", flush=True)
    t0 = time.perf_counter()
    run1_path = ROOT / "runs/training/stage5-paper-7604c2228fa94ba58dd3be0a770cf6f4/checkpoints/epoch-02"
    run1_model = AutoModelForZeroShotObjectDetection.from_pretrained(run1_path)
    run1_model.to(device)

    _, run1_raw, _ = run_single_model_inference(
        run1_model, processor, loaded_images,
        shortest=256, longest=384, device=device, desc="Stage 5 Run 1"
    )
    del run1_model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print(f"  Stage 5 Run 1 finished in {time.perf_counter() - t0:.2f}s", flush=True)

    # 3. D-095 Champion (soup 50/50, 384/576)
    print("\n[3/3] Running D-095 Champion (soup 50/50, 384/576)...", flush=True)
    t0 = time.perf_counter()
    d095_path = ROOT / "runs/training/stage5-d095-box-contrastive-20261102-a/checkpoints/best-inference"
    d095_model = AutoModelForZeroShotObjectDetection.from_pretrained(d095_path)
    d095_model.to(device)

    _, d095_raw, d095_scores = run_single_model_inference(
        d095_model, processor, loaded_images,
        shortest=384, longest=576, device=device, desc="D-095 Champion"
    )
    del d095_model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print(f"  D-095 Champion finished in {time.perf_counter() - t0:.2f}s", flush=True)

    # Clean up PIL images
    for _, pil_img in loaded_images:
        pil_img.close()

    return {
        "Baseline": base_raw,
        "Stage 5 (256/384)": run1_raw,
        "D-095 Champion (soup 50/50)": d095_raw,
        "d095_scores": d095_scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 5 D-106 Video Frame Extraction & Three-Model Detection")
    parser.add_argument(
        "--video-path",
        type=Path,
        default=ROOT / "data/input/Motion_Extract_Testing.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/video_detection/motion_extract_testing",
        help="Directory to save extracted frames and replay artifacts",
    )
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Frames sampled per second (default: 1.0)")
    parser.add_argument("--max-frames", type=int, default=30, help="Maximum frames to extract (default: 30)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument(
        "--input-image",
        type=Path,
        default=ROOT / "data/input/For_testing.jpg",
        help="Optional path to standalone input image (default: data/input/For_testing.jpg)",
    )
    args = parser.parse_args()

    frames_dir = args.output_dir / "frames"
    print(f"=== BANTAI Stage 5 D-106 Video Extraction & Detection ===")
    print(f"Video Source: {args.video_path}")
    print(f"Output Directory: {args.output_dir}")

    # Step 1: Video Extraction
    video_meta, frame_entries = extract_video_frames(
        args.video_path,
        frames_dir,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
    )
    print(f"Video Properties:")
    print(f"  Resolution: {video_meta['width']}x{video_meta['height']}")
    print(f"  FPS: {video_meta['fps']}, Total Frames: {video_meta['total_frames']}")
    print(f"  Duration: {video_meta['duration_sec']}s, Size: {video_meta['file_size_bytes'] / (1024*1024):.2f} MB")
    print(f"  SHA-256: {video_meta['sha256']}")
    print(f"Extracted {len(frame_entries)} frames into {frames_dir}")

    if args.input_image and args.input_image.is_file():
        import shutil
        img_name = args.input_image.name
        dest_p = frames_dir / img_name
        if not dest_p.is_file() or dest_p.stat().st_size != args.input_image.stat().st_size:
            shutil.copy2(args.input_image, dest_p)
        with Image.open(args.input_image) as im:
            iw, ih = im.size
        img_entry = {
            "id": 10000,
            "file_name": img_name,
            "width": iw,
            "height": ih,
            "frame_index": 0,
            "timestamp_sec": 0.0,
            "file_sha256": _sha256(dest_p),
            "media_type": "still_image",
        }
        frame_entries = [img_entry] + frame_entries
        print(f"Pre-pended standalone input image {img_name} as ID 10000 ({iw}x{ih})")

    # Step 2: Three-Model Stage 5 Inference
    inference_results = execute_three_model_detection(frame_entries, frames_dir, device=args.device)

    # Step 3: Consensus and Spatial Grouping
    print("\n=== Running D-105 Three-Model Consensus & Spatial Grouping ===")
    frames_summary = {}
    consensus_preds_by_img = {}

    for entry in frame_entries:
        img_id = entry["id"]
        d095_rows = inference_results["D-095 Champion (soup 50/50)"][img_id]
        d095_svec = inference_results["d095_scores"][img_id]
        base_rows = inference_results["Baseline"][img_id]
        run1_rows = inference_results["Stage 5 (256/384)"][img_id]

        c_rows, changed_qids, suppressed_qids = apply_scene_consensus(
            d095_rows,
            d095_svec,
            image_width=entry["width"],
            image_height=entry["height"],
            baseline_rows=base_rows,
            run1_rows=run1_rows,
        )
        consensus_preds_by_img[img_id] = c_rows

        # Visible items at calibrated video threshold >= 0.15 with containment NMS (top-25)
        visible_detections = visible_rows(
            c_rows,
            scheme="9-specific",
            nms_mode="agnostic_iou_0.50_containment_0.80",
            threshold=0.15,
        )[:25]
        grp_result = compute_spatial_grouping(
            visible_detections,
            image_width=entry["width"],
            image_height=entry["height"],
        )
        s = grp_result["summary"]

        frames_summary[str(img_id)] = {
            "file_name": entry["file_name"],
            "frame_index": entry["frame_index"],
            "timestamp_sec": entry["timestamp_sec"],
            "visible_items_count": len(visible_detections),
            "detections": [
                {
                    "category_id": r["category_id"],
                    "category_name": LABELS[r["category_id"]],
                    "bantai_category": CANONICAL_CATEGORIES.get(r["category_id"], "Other"),
                    "score": round(float(r["score"]), 4),
                    "bbox": [round(c, 2) for c in r["bbox"]],
                    "spatial_group_id": r.get("spatial_group_id"),
                    "spatial_category": r.get("spatial_category"),
                    "spatial_label": r.get("spatial_label"),
                }
                for r in grp_result["annotated_rows"]
            ],
            "spatial_grouping": {
                "total_regions": s["total_regions"],
                "garbage_piles": s["garbage_pile_count"],
                "groups_of_litter": s["group_of_individual_litter_count"],
                "individual_litter": s["individual_litter_count"],
            },
        }

        det_desc = ", ".join(
            f"{CANONICAL_CATEGORIES.get(r['category_id'])} {r['score']:.2f}"
            for r in visible_detections
        )
        reg_desc = f"[{s['total_regions']} regions: {s['garbage_pile_count']} piles, {s['group_of_individual_litter_count']} groups, {s['individual_litter_count']} indiv]"
        status_tag = f"TRASH ({len(visible_detections)})" if visible_detections else "CLEAN"
        print(f"Frame {entry['frame_index']:04d} (t={entry['timestamp_sec']:04.1f}s) ID {img_id}: {status_tag} | {reg_desc} | {det_desc if det_desc else 'None'}")

    # Step 4: Save Replay Artifacts
    showcase_path = args.output_dir / "video-showcase.json"
    vectors_path = args.output_dir / "video-score-vectors.npz"
    manifest_path = args.output_dir / "manifest.json"

    # Save video-showcase.json
    showcase_data = {
        "schema_version": "bantai.stage5-video-showcase.v1",
        "authorization": "D-112",
        "video_source": video_meta,
        "dataset_root": str(frames_dir.relative_to(ROOT)).replace("\\", "/"),
        "image_ids": [e["id"] for e in frame_entries],
        "images": {str(e["id"]): e for e in frame_entries},
        "predictions": {
            "Baseline": {str(k): v for k, v in inference_results["Baseline"].items()},
            "Stage 5 (256/384)": {str(k): v for k, v in inference_results["Stage 5 (256/384)"].items()},
            "D-095 Champion (soup 50/50)": {str(k): v for k, v in inference_results["D-095 Champion (soup 50/50)"].items()},
            "D-095 + Consensus": {str(k): v for k, v in consensus_preds_by_img.items()},
        },
    }

    with open(showcase_path, "w", encoding="utf-8") as f:
        json.dump(showcase_data, f, indent=2)
    print(f"\nSaved video showcase: {showcase_path}")

    # Save video-score-vectors.npz
    sorted_ids = [e["id"] for e in frame_entries]
    score_matrix = np.stack([inference_results["d095_scores"][i] for i in sorted_ids], axis=0)
    np.savez_compressed(
        vectors_path,
        image_ids=np.array(sorted_ids, dtype=np.int64),
        scores=score_matrix,
    )
    print(f"Saved score vectors: {vectors_path} (shape={score_matrix.shape})")

    # Save manifest.json
    manifest_data = {
        "schema_version": "bantai.stage5-video-manifest.v1",
        "authorization": "D-112",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "video_source": video_meta,
        "extracted_frames_count": len(frame_entries),
        "files": {
            "video-showcase.json": _sha256(showcase_path),
            "video-score-vectors.npz": _sha256(vectors_path),
        },
        "frames_summary": frames_summary,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved manifest: {manifest_path}")

    print("\n=== Video Extraction & Three-Model Stage 5 Detection Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
