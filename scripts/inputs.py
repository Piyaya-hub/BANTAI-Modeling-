"""Read source bytes once; orient/convert only in memory. Never save images."""

import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .configuration import ROOT

SUPPORTED = {".jpg", ".jpeg", ".png"}


class InputError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def discover_inputs(folder):
    folder = Path(folder)
    if not folder.exists():
        raise InputError("missing_input_folder", f"Input folder does not exist: {folder}")
    if not folder.is_dir():
        raise InputError("not_input_folder", f"Input path must be a directory: {folder}")
    try:
        files = sorted((p for p in folder.iterdir() if p.is_file()), key=lambda p: p.name)
    except OSError as exc:
        raise InputError("input_folder_unreadable", str(exc)) from exc
    if not files:
        raise InputError("empty_input_folder", f"No immediate files in {folder}; add JPEG/PNG images")
    return files


def source_metadata(path, root=ROOT):
    path = Path(path).resolve()
    # External configured folders use an explicit absolute identity.
    image_id = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()
    return {"image_id": image_id, "sha256": None, "width": None, "height": None,
            "coordinate_frame": "exif_oriented_original"}


def load_image(path, *, root=ROOT):
    source = source_metadata(path, root)
    if Path(path).suffix.lower() not in SUPPORTED:
        return None, source, {"stage": "input", "code": "unsupported_file", "message": "Only JPEG/PNG supported"}
    try:
        data = Path(path).read_bytes()
        source["sha256"] = hashlib.sha256(data).hexdigest()
    except OSError as exc:
        return None, source, {"stage": "input", "code": "source_read_error", "message": str(exc)}
    try:
        with Image.open(BytesIO(data)) as original:
            if original.format not in ("JPEG", "PNG"):
                raise ValueError("content is not JPEG/PNG")
            oriented = ImageOps.exif_transpose(original)
            try:
                rgb = oriented.convert("RGB")
                rgb.load()
            finally:
                if oriented is not original:
                    oriented.close()
        source.update(width=rgb.width, height=rgb.height)
        return rgb, source, None
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        return None, source, {"stage": "decode", "code": "decode_error", "message": str(exc)}
