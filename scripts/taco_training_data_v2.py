"""Read-only Stage 5 9-class COCO data loader and explicit rights gate.

Guards test.json and test photos from being loaded or read.
Supports the 9-class waste taxonomy including paper (Option A).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from .configuration import ROOT, strict_load
from .taco_training_labels_v2 import (
    LABELS,
    PROMPT,
    TAXONOMY_V2_PATH,
    digest,
    validate_taxonomy,
    validate_targets,
)
from .taco_dataset import bbox_issues, valid_id, image_path


class RightsRequired(ValueError):
    """No real-image bytes may be consumed before this gate succeeds."""


def dataset_context(folder):
    """Metadata only; deliberately never load test.json or any photographs."""
    folder = Path(folder).resolve()
    manifest = strict_load(folder / "preparation_manifest.json")
    if manifest["schema_version"] != "bantai.taco-preparation.v1" or manifest["status"] != "validated":
        raise ValueError("Require validated Stage 5 export")
    observed_source = digest(manifest["source"]["annotations"])
    serialization_only = False
    if observed_source != manifest["source"]["sha256"]:
        reconstructed = json.dumps(strict_load(Path(manifest["source"]["annotations"]))).encode("utf-8")
        if hashlib.sha256(reconstructed).hexdigest() != manifest["source"]["sha256"]:
            raise ValueError("Original source annotation JSON changed")
        serialization_only = True
    validate_taxonomy()
    if digest(TAXONOMY_V2_PATH) != manifest["source"]["taxonomy_sha256"]:
        raise ValueError("Taxonomy v2 changed")
    context = {
        "manifest_sha256": digest(folder / "preparation_manifest.json"),
        "source_annotations_sha256": manifest["source"]["sha256"],
        "splits": {},
    }
    if serialization_only:
        context.update(
            observed_source_annotations_sha256=observed_source,
            source_verification="exact_recorded_bytes_reconstructed_by_json_dumps",
        )
    seen = set()
    for split in ("train", "validation"):
        path = folder / (split + ".json")
        if digest(path) != manifest["artifacts"][path.name]:
            raise ValueError("Split hash mismatch: " + split)
        data = strict_load(path)
        ids = [i["id"] for i in data["images"]]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Duplicate or overlapping train/validation image IDs")
        seen.update(ids)
        context["splits"][split] = {
            "sha256": digest(path),
            "image_ids": ids,
            "images": len(ids),
            "annotations": len(data["annotations"]),
        }
    return context


def require_rights(path, context):
    """Validate owner-supplied rights record."""
    if path is None or not Path(path).is_file():
        raise RightsRequired("Documented photo AND annotation/database authorization is required")
    record = strict_load(Path(path))
    if record.get("dataset") != context or record.get("scope") != "local_academic_train_validation_no_publication":
        raise RightsRequired("Rights record is incomplete or does not cover this exact train/validation snapshot")
    if record.get("schema_version") == "bantai.training-rights.v2":
        if (
            record.get("basis_type") != "owner_authorized_section185_research"
            or record.get("workflow_authorized") is not True
            or record.get("basis_assessment") != "defensible_limited_use_with_uncertainty"
            or any(
                record.get(k) != "not_claimed"
                for k in ("rightsholder_permission", "institutional_approval", "definitive_legal_clearance")
            )
            or any(record.get(k) is not False for k in ("photo_rights_cleared", "annotation_database_rights_cleared"))
            or record.get("owner_declaration")
            != {
                "non_commercial_academic": True,
                "local_only": True,
                "no_dataset_publication_redistribution_upload": True,
                "workflow_authorized": True,
            }
            or set(record.get("factors", {})) != {"purpose", "nature", "amount", "market"}
            or not all(isinstance(v, str) and v.strip() for v in record["factors"].values())
        ):
            raise RightsRequired("Owner-authorized exception record is incomplete or misrepresents clearance")
        assessment = record.get("assessment", {})
        doc_path = (ROOT / assessment.get("path", "")).resolve()
        expected = assessment.get("sha256", "")
        if not doc_path.is_relative_to(ROOT / "docs"):
            raise RightsRequired("Exception assessment path escapes docs")
        if not doc_path.is_file() or digest(doc_path) != expected:
            # Versioned updates preserve the exact earlier assessment, not a new basis.
            if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                raise RightsRequired("Invalid assessment digest")
            archived = ROOT / "docs" / "assessments" / (expected + ".md")
            if not archived.is_file() or digest(archived) != expected:
                raise RightsRequired("Exception assessment evidence is missing or changed")
    else:
        raise RightsRequired("Unsupported rights record schema")
    return {"record_sha256": digest(path), "record": record}


class CocoTrainingDatasetV2:
    """Stage 5 9-class dataset loader."""

    def __init__(self, folder, split, rights_path):
        if split not in ("train", "validation"):
            raise ValueError("Test split is deliberately unavailable to the Stage 5 trainer")
        self.context = dataset_context(folder)
        self.rights = require_rights(rights_path, self.context)
        self.data = strict_load(Path(folder) / (split + ".json"))
        if [(c["id"], c["name"]) for c in self.data["categories"]] != list(enumerate(LABELS)):
            raise ValueError("COCO category order differs from approved 9-class taxonomy")
        self.root = Path(self.data["info"]["image_root"]).resolve()
        self.photo_hashes = self.rights["record"].get("photo_sha256", {})
        if self.photo_hashes:
            expected_ids={str(i) for value in self.context['splits'].values() for i in value['image_ids']}
            if set(self.photo_hashes)!=expected_ids:raise RightsRequired('Incomplete source-photo hash binding')
            for image in self.data['images']:
                proof=self.photo_hashes[str(image['id'])]
                if proof['split']!=split or proof['file_name']!=image['file_name'] or proof['sha256']!=image['source_taco']['file_sha256']:
                    raise RightsRequired('Photo binding differs from export')
        self.images = self.data["images"]
        self.photo_reads = 0
        self.objects = {i["id"]: [] for i in self.images}
        image_map = {i["id"]: i for i in self.images}
        seen = set()
        for ann in self.data["annotations"]:
            image = image_map.get(ann.get("image_id"))
            if image is None or not valid_id(ann.get("id")) or ann["id"] in seen:
                raise ValueError("Invalid COCO image reference or annotation ID")
            seen.add(ann["id"])
            if (
                type(ann.get("category_id")) is not int
                or not 0 <= ann["category_id"] < len(LABELS)
                or ann.get("iscrowd") != 0
                or ann.get("ignore", 0) != 0
                or bbox_issues(ann.get("bbox"), image["width"], image["height"])
                or type(ann.get("area")) not in (int, float)
                or not math.isfinite(ann["area"])
                or ann["area"] <= 0
            ):
                raise ValueError("Invalid/ignored/excluded annotation; no silent dropping")
            self.objects[ann["image_id"]].append(ann)
        if any(not rows for rows in self.objects.values()):
            raise ValueError("Unexpected empty image in strict-clean stage 5")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        from PIL import Image

        row = self.images[index]
        path = image_path(self.root, row["file_name"])
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f"Unsafe or missing source photograph: {path}")
        if self.photo_hashes and digest(path)!=self.photo_hashes[str(row["id"])]["sha256"]:
            raise ValueError("Source-photo SHA-256 changed")
        self.photo_reads += 1
        with Image.open(path) as image:
            image.load()
            if image.size != (row["width"], row["height"]) or image.getexif().get(274, 1) != 1:
                raise ValueError("Actual photograph dimensions/orientation differ from strict-clean coordinates")
            rgb = image.convert("RGB")
        return rgb, {"image_id": row["id"], "annotations": copy.deepcopy(self.objects[row["id"]])}


def collate(processor, samples, shortest=256, longest=384):
    """Explicit fixed square padding keeps 900 proposals valid for narrow photos."""
    import torch
    from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps
    from .training_smoke import validate_encoder_capacity, valid_image_loss_targets

    images, targets = zip(*samples)
    batch = processor(
        images=list(images),
        text=[PROMPT] * len(images),
        annotations=copy.deepcopy(list(targets)),
        padding=True,
        return_tensors="pt",
        size={"shortest_edge": shortest, "longest_edge": longest},
        pad_size={"height": longest, "width": longest},
    )
    ph, pw = batch["pixel_values"].shape[-2:]
    validate_encoder_capacity(ph, pw)
    maps = build_label_maps(torch.zeros(len(images), 1, 256), batch["input_ids"])
    geometry = []
    for i, (image, target, label, mapping) in enumerate(zip(images, targets, batch["labels"], maps)):
        spans = [
            processor.tokenizer.decode(batch["input_ids"][i][m[: len(batch["input_ids"][i])].bool()])
            for m in mapping
        ]
        if spans != [name.replace("_", " ") for name in LABELS]:
            raise ValueError(f"Prompt/category span mismatch: {spans} vs {[n.replace('_', ' ') for n in LABELS]}")
        mask = batch["pixel_mask"][i]
        rw, rh = int(mask[0].sum()), int(mask[:, 0].sum())
        valid_image_loss_targets(label, target, image.size, (rw, rh), (pw, ph))
        geometry.append({
            "image_id": target["image_id"],
            "original_wh": list(image.size),
            "resized_wh": [rw, rh],
            "padded_wh": [pw, ph],
        })
    return batch, geometry


def get_batch(dataset, indexes, processor, shortest, longest):
    samples = []
    try:
        for i in indexes:
            samples.append(dataset[i])
        return collate(processor, samples, shortest, longest)
    finally:
        for image, _ in samples:
            image.close()
