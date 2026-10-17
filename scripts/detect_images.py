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
    parser = argparse.ArgumentParser(description="BANTAI: preparation by default; explicit inference and native TkAgg viewing")
    parser.add_argument("--mode", choices=("prepare", "infer", "replay", "display-check"), default="prepare")
    parser.add_argument("--config", default="config/detection.json")
    parser.add_argument("--input", help="override input directory; relative to repository root")
    parser.add_argument("--results", help="override JSON output directory; relative to repository root")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="override requested device")
    parser.add_argument("--no-display", action="store_true", help="disable the native image window; print tables only")
    parser.add_argument("--local-files-only", action="store_true", help="inference may use cached files only; never contact the Hub")
    parser.add_argument("--terminal-mode", choices=("summary", "full"), help="with --no-display: summary passing-only or full raw diagnostics; native viewing always uses passing-only tables")
    parser.add_argument("--summary-below-threshold", type=int, help="legacy option, ignored by passing-only console output")
    parser.add_argument("--page-rows", type=int, help="maximum physical data lines per table page (1..100; default 10)")
    parser.add_argument("--include-special-tokens", action="store_true", help="full mode: include active special-token diagnostics")
    parser.add_argument("--preview-width", type=int, help="legacy terminal option; ignored by the native image viewer")
    parser.add_argument("--paginate", action=argparse.BooleanOptionalAction, default=None, help="legacy option; native viewer closure controls continuation")
    parser.add_argument("--result", help="replay: existing functional result JSON (no model execution)")
    parser.add_argument("--image", help="replay: matching unchanged original JPEG/PNG")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, overrides={"input": args.input, "results": args.results})
        if args.terminal_mode:
            config["display"]["terminal_mode"] = args.terminal_mode
        for key in ("summary_below_threshold", "page_rows", "preview_width", "paginate"):
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
        if args.mode != "prepare" and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if args.mode == "display-check":
            if args.no_display:
                raise ValueError("display-check requires display; use replay --no-display to verify source identity without rendering")
            from scripts.viewer import show_synthetic, NativeDisplayError
            try:
                show_synthetic(preview_width=config["display"]["preview_width"])
            except NativeDisplayError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            except Exception as exc:
                print(f"Synthetic renderer failed ({type(exc).__name__}): {exc}",file=sys.stderr)
                return 1
            return 0
        if args.mode == "replay":
            if not args.result or not args.image:
                raise ValueError("--mode replay requires --result and --image")
            from scripts.replay import run_replay
            return run_replay(config,args.result,args.image)
        files = discover_inputs(config["paths"]["input"])
        if args.mode == "infer":
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
        print(f"{args.mode.capitalize()} failed [{getattr(exc, 'code', type(exc).__name__)}]: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
