#!/usr/bin/env python3
"""
Recover missing TACO dataset images from Flickr.

Resolves missing images by reading annotations.json in the TACO dataset folder,
identifying files missing on disk, and downloading them directly from Flickr.

Includes resilience for ISP/DNS routing loops by routing Flickr CloudFront requests
through verified active edge IPs.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# Verified alternate CloudFront edge IPs to bypass ISP routing loops on 13.33.145.57
CLOUDFRONT_IPS = [
    "18.64.8.53",
    "99.84.181.57",
    "18.165.83.57",
    "143.204.101.57",
    "108.138.16.57",
]

_active_ip_index = 0
_orig_getaddrinfo = socket.getaddrinfo


def _patched_getaddrinfo(host: str, port: int, family: int = 0, type: int = 0, proto: int = 0, flags: int = 0) -> Any:
    global _active_ip_index
    if "staticflickr.com" in host:
        ip = CLOUDFRONT_IPS[_active_ip_index % len(CLOUDFRONT_IPS)]
        return _orig_getaddrinfo(ip, port, socket.AF_INET, type, proto, flags)
    return _orig_getaddrinfo(host, port, family, type, proto, flags)


def cycle_cloudfront_ip() -> str:
    global _active_ip_index
    _active_ip_index = (_active_ip_index + 1) % len(CLOUDFRONT_IPS)
    return CLOUDFRONT_IPS[_active_ip_index]


def install_dns_patch() -> None:
    socket.getaddrinfo = _patched_getaddrinfo


def test_connectivity() -> bool:
    """Verify that the selected CloudFront IP can reach Flickr over HTTPS."""
    import requests

    for _ in range(len(CLOUDFRONT_IPS)):
        try:
            r = requests.get(
                "https://farm66.staticflickr.com/65535/46939445965_86fdbb614c_o.png",
                timeout=8,
                headers={"User-Agent": "Mozilla/5.0 (BANTAI TACO Recovery)"},
            )
            if r.status_code == 200:
                return True
        except Exception:
            cycle_cloudfront_ip()
    return False


def recover_missing_images(
    dataset_dir: Path,
    dry_run: bool = False,
    max_downloads: Optional[int] = None,
    delay: float = 0.2,
) -> int:
    import requests
    from PIL import Image

    annotations_file = dataset_dir / "annotations.json"
    if not annotations_file.is_file():
        print(f"ERROR: Annotations file not found: {annotations_file}", file=sys.stderr)
        return 1

    print(f"Loading annotations from {annotations_file}...")
    with open(annotations_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    images: List[Dict[str, Any]] = data.get("images", [])
    print(f"Total images registered in annotations: {len(images)}")

    missing: List[Dict[str, Any]] = []
    for img in images:
        file_name = img["file_name"]
        local_path = dataset_dir / file_name
        if not local_path.is_file():
            missing.append(img)

    print(f"Missing images on disk: {len(missing)} / {len(images)}")

    if not missing:
        print("All images already exist on disk! No downloads required.")
        return 0

    # Summary by batch
    batch_counts: Dict[str, int] = {}
    for img in missing:
        batch = img["file_name"].split("/")[0] if "/" in img["file_name"] else "root"
        batch_counts[batch] = batch_counts.get(batch, 0) + 1

    print("\nMissing breakdown by batch:")
    for batch, count in sorted(batch_counts.items()):
        print(f"  {batch}: {count} missing")

    if dry_run:
        print("\nDry run requested. Exiting without downloading.")
        return 0

    to_download = missing if max_downloads is None else missing[:max_downloads]
    print(f"\nStarting download of {len(to_download)} images...")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (BANTAI TACO Recovery)"})

    downloaded = 0
    failed: List[Dict[str, str]] = []
    start_time = time.time()

    for i, img in enumerate(to_download, 1):
        file_name = img["file_name"]
        target_path = dataset_dir / file_name
        target_path.parent.mkdir(parents=True, exist_ok=True)

        url_orig = img.get("flickr_url")
        url_640 = img.get("flickr_640_url")
        expected_w = img.get("width")
        expected_h = img.get("height")

        success = False
        last_error = ""

        # Try original URL first, then 640 fallback if original fails with 404
        for url in [url_orig, url_640]:
            if not url:
                continue

            # Try up to 3 retries cycling CloudFront edge IPs
            for attempt in range(3):
                try:
                    resp = session.get(url, timeout=20)
                    if resp.status_code == 404:
                        last_error = f"HTTP 404 Not Found at {url}"
                        break  # Fall back to next URL candidate
                    resp.raise_for_status()

                    # Validate image bytes
                    img_bytes = BytesIO(resp.content)
                    with Image.open(img_bytes) as pil_img:
                        pil_img.load()
                        w, h = pil_img.size
                        save_format = pil_img.format or "JPEG"
                        exif = pil_img.info.get("exif")

                        temp_path = target_path.with_name(target_path.name + ".tmp")
                        save_kwargs = {"format": save_format}
                        if exif and save_format in ("JPEG", "TIFF", "WEBP"):
                            save_kwargs["exif"] = exif

                        pil_img.save(temp_path, **save_kwargs)
                        temp_path.replace(target_path)
                        success = True
                        downloaded += 1

                        dim_note = f"{w}x{h}"
                        if expected_w and expected_h and (w != expected_w or h != expected_h):
                            dim_note += f" (expected {expected_w}x{expected_h})"

                        print(f"[{i}/{len(to_download)}] OK: {file_name} [{dim_note}]")
                        break

                except Exception as exc:
                    last_error = str(exc)
                    cycle_cloudfront_ip()
                    time.sleep(0.5)

            if success:
                break

        if not success:
            print(f"[{i}/{len(to_download)}] FAILED: {file_name} ({last_error})")
            failed.append({"file_name": file_name, "error": last_error})

        if delay > 0:
            time.sleep(delay)

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"Download complete in {elapsed:.1f}s.")
    print(f"  Successfully recovered: {downloaded} / {len(to_download)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print("\nFailed images:")
        for f_item in failed:
            print(f"  - {f_item['file_name']}: {f_item['error']}")
        return 1

    return 0


def main() -> int:
    default_dataset = Path(r"C:\Users\Justine\Documents\Python\BANTAI-Datasets\TACO")
    parser = argparse.ArgumentParser(description="Recover missing TACO dataset images from Flickr.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset,
        help="Path to TACO dataset root directory (containing annotations.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List missing images without downloading",
    )
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=None,
        help="Limit number of images to download (useful for testing)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        help="Delay between requests in seconds (default: 0.15s)",
    )
    args = parser.parse_args()

    install_dns_patch()

    print("Verifying Flickr CDN connectivity...")
    if not test_connectivity():
        print("WARNING: Direct test connection failed. Proceeding with retries...", file=sys.stderr)
    else:
        print("Flickr CDN connectivity confirmed.")

    return recover_missing_images(
        dataset_dir=args.dataset_dir,
        dry_run=args.dry_run,
        max_downloads=args.max_downloads,
        delay=args.delay,
    )


if __name__ == "__main__":
    sys.exit(main())
