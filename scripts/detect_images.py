"""Phase 2B preparation CLI. No detector, model loading, inference or viewer."""

import argparse
import importlib.metadata
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.configuration import ConfigError, load_config
from scripts.inputs import InputError, discover_inputs, load_image
from scripts.persistence import atomic_json, create_run, source_key


def main(argv=None):
    parser = argparse.ArgumentParser(description="BANTAI preparation only: inspect JPEG/PNG inputs; no inference or viewer")
    parser.add_argument("--config", default="config/detection.json")
    parser.add_argument("--input", help="override input directory; relative to repository root")
    parser.add_argument("--results", help="override JSON output directory; relative to repository root")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, overrides={"input": args.input, "results": args.results})
        files = discover_inputs(config["paths"]["input"])
        run_id, folder = create_run(config["paths"]["results"])
        manifest = {"schema_version": "bantai.preparation-run.v1", "run_id": run_id,
                    "mode": "preparation_only", "inference_executed": False,
                    "config": config, "pillow_version": importlib.metadata.version("Pillow"),
                    "records": [], "status": "in_progress"}
        atomic_json(folder / "preparation-run.json", manifest)
        errors = 0
        prepared = 0
        for path in files:
            image, source, error = load_image(path)
            if image is not None:
                image.close()
            status = ("skipped_unsupported" if error and error["code"] == "unsupported_file"
                      else "error" if error else "prepared")
            errors += status == "error"
            prepared += status == "prepared"
            record = {"schema_version": "bantai.preparation.v1", "run_id": run_id,
                      "mode": "preparation_only", "source": source, "status": status,
                      "inference_executed": False,
                      "display": {"enabled": config["display"]["enabled"], "status": "not_attempted", "error": None},
                      "error": error}
            name = source_key(source) + ".preparation.json"
            atomic_json(folder / name, record)
            manifest["records"].append(name)
            print(f"{path.name}: {status}" + (f" ({error['code']})" if error else " — inference not run"))
        manifest.update(status="completed_with_errors" if errors else "completed", prepared=prepared, errors=errors)
        atomic_json(folder / "preparation-run.json", manifest, replace=True)
        print(f"Preparation only; {prepared} decoded, {errors} errors. JSON: {folder}")
        if not prepared and not errors:
            print("No supported JPEG/PNG inputs; unsupported files were skipped.", file=sys.stderr)
            return 2
        return 1 if errors else 0
    except (ConfigError, InputError, OSError, ValueError) as exc:
        print(f"Preparation failed [{getattr(exc, 'code', type(exc).__name__)}]: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
