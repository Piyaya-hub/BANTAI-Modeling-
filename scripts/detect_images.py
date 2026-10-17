"""BANTAI CLI: preparation by default; explicit inference uses the pinned model."""

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
    parser = argparse.ArgumentParser(description="BANTAI: preparation only by default; --mode infer runs pinned pretrained inference and optional terminal viewing")
    parser.add_argument("--mode", choices=("prepare", "infer"), default="prepare")
    parser.add_argument("--config", default="config/detection.json")
    parser.add_argument("--input", help="override input directory; relative to repository root")
    parser.add_argument("--results", help="override JSON output directory; relative to repository root")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="override requested device")
    parser.add_argument("--no-display", action="store_true", help="deliberately disable terminal image rendering for inference")
    parser.add_argument("--local-files-only", action="store_true", help="inference may use cached files only; never contact the Hub")
    parser.add_argument("--terminal-mode", choices=("summary", "full"), help="summary or every raw query/token score; complete diagnostics always saved")
    parser.add_argument("--summary-below-threshold", type=int, help="summary: show this many highest-scoring valid below-threshold queries (0..900; default 1)")
    parser.add_argument("--page-rows", type=int, help="maximum physical data lines per table page (1..100; default 10)")
    parser.add_argument("--include-special-tokens", action="store_true", help="full mode: explicitly include active special tokens in the readable table")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, overrides={"input": args.input, "results": args.results})
        if args.terminal_mode:
            config["display"]["terminal_mode"] = args.terminal_mode
        for key in ("summary_below_threshold", "page_rows"):
            value = getattr(args,key)
            if value is not None:
                config["display"][key] = value
        if args.include_special_tokens:
            config["display"]["include_special_tokens"] = True
        from scripts.configuration import validate_config
        config = validate_config(config)
        if args.device:
            config["device"] = args.device
        if args.no_display:
            config["display"]["enabled"] = False
        files = discover_inputs(config["paths"]["input"])
        if args.mode == "infer":
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            from scripts.pipeline import run_inference
            return run_inference(config, files, local_files_only=args.local_files_only)
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
