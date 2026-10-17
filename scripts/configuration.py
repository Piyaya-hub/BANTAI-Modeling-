"""Strict configuration loading. Relative paths always use the repository root."""

import copy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"


class ConfigError(ValueError):
    """Invalid or unreadable preparation configuration."""


def finite_number(value):
    return type(value) in (int, float) and (type(value) is int or math.isfinite(value))


def strict_load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"non-JSON numeric constant: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=constant)


def resolve_path(value, root=ROOT):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConfigError("paths must be nonempty strings without NUL")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def validate_config(value, root=ROOT):
    c = copy.deepcopy(value)
    try:
        if not isinstance(c, dict):
            raise ConfigError("configuration must be an object")
        for key in ("input", "results", "cache"):
            c["paths"][key] = str(resolve_path(c["paths"][key], root))
        paths = [Path(c["paths"][k]) for k in ("input", "results", "cache")]
        for i, a in enumerate(paths):
            for b in paths[i + 1:]:
                if a == b or a in b.parents or b in a.parents:
                    raise ConfigError("input, results and cache paths must not overlap")
        m = c["model"]
        if (m["id"] != MODEL_ID or m["revision"] != REVISION
                or m["processor_revision"] != REVISION):
            raise ConfigError("use the 2A-verified checkpoint and same processor revision")
        if m["use_safetensors"] is not True or m["disable_custom_kernels"] is not True:
            raise ConfigError("safetensors and disabled custom kernels are required")
        for key in ("threshold", "text_threshold"):
            if not finite_number(c[key]) or not 0 <= c[key] <= 1:
                raise ConfigError(f"{key} must be a finite number in [0,1]")
        if not isinstance(c["prompt"], str) or not c["prompt"].strip():
            raise ConfigError("prompt must be nonempty")
        if c["device"] not in ("auto", "cpu", "cuda") or c["dtype"] != "float32":
            raise ConfigError("device must be auto/cpu/cuda; dtype must be float32")
        d = c["display"]
        if type(d["enabled"]) is not bool or d["backend"] != "TkAgg" or d["save_annotated"] is not False:
            raise ConfigError("display requires boolean enabled, TkAgg, save_annotated=false")
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"missing or malformed configuration field: {exc}") from exc
    return c


def load_config(path="config/detection.json", *, root=ROOT, overrides=None):
    try:
        c = strict_load(resolve_path(str(path), root))
        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    c["paths"][key] = str(value)
        return validate_config(c, root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigError(f"configuration: {exc}") from exc
