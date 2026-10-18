"""Strict JSON publication using same-directory temporary files; no image output."""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid


def sanitize_numbers(value, diagnostics=None, field="$"):
    """Replace nonfinite metadata recursively, recording paths without mutating input."""
    if diagnostics is None:
        diagnostics = []
    if isinstance(value, float) and not math.isfinite(value):
        diagnostics.append({"field": field, "reason": "nonfinite_number"})
        return None
    if isinstance(value, dict):
        return {k: sanitize_numbers(v, diagnostics, f"{field}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_numbers(v, diagnostics, f"{field}[{i}]") for i, v in enumerate(value)]
    return value


def strict_payload(record):
    record = copy.deepcopy(record)
    diagnostics = []
    record = sanitize_numbers(record, diagnostics)
    if diagnostics:
        record.setdefault("numeric_diagnostics", []).extend(diagnostics)
    return json.dumps(record, allow_nan=False, ensure_ascii=False, indent=2) + "\n"


def replace_atomic(temp, path):
    """Bound transient Windows access/sharing retries; never delete the old file."""
    for attempt, delay in enumerate((0, .05, .1, .2, .4)):
        if delay:
            time.sleep(delay)
        try:
            os.replace(temp, path)
            return
        except PermissionError as exc:
            if os.name != 'nt' or getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 4:
                raise


def atomic_json(path, record, *, replace=False):
    """Create without clobbering; explicit replacement preserves prior file on failure.

    Hard-link publication is atomic and fails if an initial destination exists.
    Filesystems without hard-link support fail visibly rather than weaken safety.
    os.replace handles explicit updates, with bounded Windows access/sharing
    retries (at most 0.75 seconds). Flush/fsync precede either publication.
    Atomic visibility is guaranteed; power-loss directory durability is not claimed.
    """
    path = Path(path)
    payload = strict_payload(record)  # Serialization failures must leave no temp file.
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=".bantai-", suffix=".tmp",
                                         delete=False) as out:
            temp = Path(out.name)
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        if replace:
            replace_atomic(temp, path)
        else:
            os.link(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def create_run(results):
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        run_id = uuid.uuid4().hex
        folder = results / run_id
        try:
            folder.mkdir()
            return run_id, folder
        except FileExistsError:
            continue
    raise FileExistsError("Could not reserve a unique run directory")


def source_key(source):
    # Include full identity AND content, never a basename or stem alone.
    return hashlib.sha256(json.dumps([source["image_id"], source["sha256"]],
                                    ensure_ascii=False).encode()).hexdigest()
