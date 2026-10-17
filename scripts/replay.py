"""Display saved predictions without importing or executing the model."""

import copy
import hashlib
from pathlib import Path
import sys

from .configuration import resolve_path, strict_load
from .inputs import load_image
from .persistence import atomic_json, create_run
from .results import validate_result
from .viewer import TerminalDisplayError, safe_text, show_image, summary


def run_replay(config, result_path, image_path):
    result_path = resolve_path(str(result_path))
    image_path = resolve_path(str(image_path))
    before = result_path.read_bytes()
    result = validate_result(strict_load(result_path))
    if result_path.read_bytes() != before:
        raise ValueError("Result JSON changed during validation; retry with a stable result")
    image, source, error = load_image(image_path)
    if image is None:
        raise ValueError(f"Replay original cannot be decoded: {error['code']}")
    try:
        # Same bytes at another path are not the recorded source identity.
        for key in ("image_id", "sha256", "width", "height", "coordinate_frame"):
            if source[key] != result["source"][key]:
                raise ValueError(f"Replay source mismatch: {key}; use the matching unchanged original")
        shown = copy.deepcopy(result)
        shown["display"].update(config["display"], console_policy="passing_only")
        run_id, folder = create_run(config["paths"]["results"])
        path = folder / "replay.json"
        record = {"schema_version": "bantai.display-replay.v1", "run_id": run_id,
                  "inference_executed": False, "uses_saved_predictions": True,
                  "source_result": {"path": str(result_path), "sha256": hashlib.sha256(before).hexdigest(),
                                    "run_id": result["run_id"]},
                  "source": source, "model_from_saved_result": result["model"],
                  "thresholds_from_saved_result": {k: result["inference"][k] for k in ("threshold", "text_threshold")},
                  "display": {"enabled": config["display"]["enabled"],
                              "status": "pending" if config["display"]["enabled"] else "disabled",
                              "error": None, "visual_verification": "pending", "console_policy": "passing_only"}}
        atomic_json(path, record)
        try:
            if config["display"]["enabled"]:
                try:
                    show_image(image, shown)
                except Exception as exc:
                    record["display"].update(status="error", error={
                        "stage": "display", "code": "unsupported_terminal" if isinstance(exc, TerminalDisplayError) else "viewer_failed",
                        "message": str(exc) if isinstance(exc, TerminalDisplayError) else f"Replay rendering/write failed ({type(exc).__name__})"})
                    atomic_json(path, record, replace=True)
                    print(f"Replay display failed: {safe_text(record['display']['error']['message'])}", file=sys.stderr)
                    print(summary(shown))
                else:
                    record["display"]["status"] = "shown"
                    atomic_json(path, record, replace=True)
            else:
                print(f"IMAGE: {safe_text(Path(source['image_id']).name)}")
                print("Image display deliberately disabled; replay verifies source identity only.")
                print(summary(shown))
        except KeyboardInterrupt:
            print(f"Replay interrupted; saved pending display remains unfinished: {path}", file=sys.stderr)
            return 130
        print(f"Display-only replay JSON: {path}")
        return 1 if record["display"]["status"] == "error" else 0
    finally:
        image.close()
