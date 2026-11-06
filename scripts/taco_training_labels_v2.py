"""Stage 5 nine-class waste taxonomy, prompt and label validation."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys

from .configuration import MODEL_ID, REVISION, ROOT, strict_load

LABELS = (
    "cigarette",
    "plastic_bottle",
    "plastic_bag",
    "plastic_wrapper",
    "plastic_film",
    "drink_can",
    "plastic_cup",
    "styrofoam_piece",
    "paper",
)

PROMPT = ". ".join(label.replace("_", " ") for label in LABELS) + "."
SNAPSHOT = ROOT / ".cache/huggingface" / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / REVISION
TAXONOMY_V2_PATH = ROOT / "config/taco_taxonomy.v2.json"


def digest(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_taxonomy():
    taxonomy = strict_load(TAXONOMY_V2_PATH)
    if [(r["id"], r["name"]) for r in taxonomy["labels"]] != list(enumerate(LABELS)):
        raise ValueError("Approved nine-class taxonomy/order changed")
    return taxonomy


def validate_targets(images, targets):
    if not images or len(images) != len(targets):
        raise ValueError("Image/target batch mismatch")
    for image, target in zip(images, targets):
        for obj in target["annotations"]:
            label, box = obj["category_id"], obj["bbox"]
            if type(label) is not int or not 0 <= label < len(LABELS):
                raise ValueError(f"Invalid target class ID: {label} (expected 0..{len(LABELS) - 1})")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError(f"Invalid bbox format: {box}")
            w, h = box[2], box[3]
            if w <= 0 or h <= 0:
                raise ValueError(f"Non-positive bbox dimension: {box}")
