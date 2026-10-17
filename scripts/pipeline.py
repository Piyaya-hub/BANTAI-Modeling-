"""Functional inference orchestration: persist before display, then save its outcome."""

import copy
from pathlib import Path
import platform
import sys

from .inference import GroundingDinoDetector, InferenceError, package_snapshot
from .inputs import load_image
from .persistence import atomic_json, create_run
from .postprocessing import DUPLICATE_POLICY, MAPPING_VERSION, apply_policies
from .results import make_result, prepare_result, save_display_outcome, save_result
from .viewer import TerminalDisplayError, safe_text, show_image, summary


def run_inference(config, files, *, local_files_only=False):
    run_id, folder = create_run(config["paths"]["results"])
    manifest = {"schema_version": "bantai.inference-run.v1", "run_id": run_id,
                "mode": "pretrained_inference", "config": config,
                "local_files_only": local_files_only, "records": [], "status": "in_progress",
                "processing_errors": 0, "display_errors": 0, "persistence_errors": 0,
                "inference_completed_images": 0, "skipped_unsupported": 0}
    detector = None
    announced_fallbacks = 0
    startup_error = None
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": package_snapshot(), "model_loaded": False, "inference_attempted": False,
                   "inference_completed_images": 0}
    try:
        atomic_json(folder / "run.json", manifest)
        atomic_json(folder / "environment.json", environment)
        for path in files:
            image = None
            try:
                image, source, input_error = load_image(path)
                execution, predictions, timing, flags = None, [], None, []
                library_version = None
                error = input_error
                unsupported = input_error is not None and input_error["code"] == "unsupported_file"
                completed = False
                if image is not None:
                    if detector is None and startup_error is None:
                        try:
                            detector = GroundingDinoDetector(config, local_files_only=local_files_only)
                        except InferenceError as exc:
                            startup_error = exc.record()
                    if startup_error is not None:
                        error = startup_error
                    else:
                        library_version = detector.versions["transformers"]
                        execution = {"actual_device": detector.device,
                                     "environment_manifest": f"{run_id}/environment.json",
                                     "preprocessing_manifest": f"{run_id}/preprocessing.json",
                                     "mapping_version": MAPPING_VERSION, "duplicate_policy": DUPLICATE_POLICY,
                                     "fallback": copy.deepcopy(detector.fallbacks) or None}
                        try:
                            predictions, timing = detector.predict(image)
                            predictions, flags = apply_policies(predictions, overlap_iou=config["postprocessing"]["overlap_iou"])
                            completed = True
                        except InferenceError as exc:
                            error = exc.record()
                        # An OOM retry can change actual device and provenance.
                        execution.update(actual_device=detector.device, fallback=copy.deepcopy(detector.fallbacks) or None)
                        for fallback in detector.fallbacks[announced_fallbacks:]:
                            print(f"Device fallback: {fallback['from']} -> {fallback['to']} ({fallback['stage']}/{fallback['code']})")
                        announced_fallbacks = len(detector.fallbacks)
                        environment = detector.environment()
                        environment.update(model_loaded=True, inference_attempted=True,
                                           inference_completed_images=manifest["inference_completed_images"] + int(completed))
                        atomic_json(folder / "environment.json", environment, replace=True)
                        atomic_json(folder / "preprocessing.json", detector.preprocessing,
                                    replace=(folder / "preprocessing.json").exists())
                result = make_result(run_id, source, config, detections=predictions,
                                     inference_completed=completed, execution=execution,
                                     error=error, unsupported=unsupported, library_version=library_version)
                if timing is not None:
                    result["timing"] = timing
                if completed and isinstance(getattr(detector, "query_diagnostics", None), dict):
                    result["query_diagnostics"] = copy.deepcopy(detector.query_diagnostics)
                result["display"].update({key: config["display"][key] for key in ("backend", "terminal_mode", "summary_below_threshold", "page_rows", "include_special_tokens")})
                result["review"]["flags"] = flags
                result = prepare_result(result)
                # Initial failure exits before summary/viewer. Previous records remain saved.
                result_path = save_result(folder, result)
                manifest["records"].append(result_path.name)
                manifest["processing_errors"] += result["status"] == "error"
                manifest["skipped_unsupported"] += unsupported
                manifest["inference_completed_images"] += completed
                atomic_json(folder / "run.json", manifest, replace=True)
                if result["display"]["status"] == "pending":
                    try:
                        show_image(image, result, cache_dir=config["paths"]["cache"])
                    except Exception as exc:
                        display_error = {"stage": "display", "code": "unsupported_terminal" if isinstance(exc, TerminalDisplayError) else "viewer_failed",
                                         "message": str(exc) if isinstance(exc, TerminalDisplayError) else f"Terminal rendering/write failed ({type(exc).__name__}); inference results remain saved"}
                        manifest["display_errors"] += 1
                        save_display_outcome(result_path, "error", error=display_error)
                        print(f"Display failed: {safe_text(path.name)}; {display_error['message']}; inference JSON preserved", file=sys.stderr)
                        print(summary(result, mode=config["display"]["terminal_mode"]))
                    else:
                        save_display_outcome(result_path, "shown")
                else:
                    print(summary(result, mode=config["display"].get("terminal_mode", "summary")))
            finally:
                if image is not None:
                    image.close()
        manifest["status"] = "completed_with_errors" if manifest["processing_errors"] or manifest["display_errors"] else "completed"
        atomic_json(folder / "run.json", manifest, replace=True)
    except (OSError, ValueError) as exc:
        manifest["persistence_errors"] += 1
        manifest["status"] = "interrupted_by_output_error"
        # A failed metadata update must not replace/erase an earlier saved result.
        print(f"Result/metadata persistence failed ({type(exc).__name__}): {exc}; inspect {folder}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"Interrupted; saved pending display outcomes remain unfinished: {folder}", file=sys.stderr)
        return 130
    finally:
        if detector is not None:
            detector.close()
    print(f"Inference run JSON: {folder}")
    if not manifest["inference_completed_images"] and not manifest["processing_errors"]:
        print("No supported decodable inputs; inference was not run.", file=sys.stderr)
        return 2
    return 1 if manifest["processing_errors"] or manifest["display_errors"] else 0
