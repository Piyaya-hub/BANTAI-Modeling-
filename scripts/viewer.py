"""Native in-memory viewing and summaries. Never save an image or start a server."""

from collections import Counter
from pathlib import Path
import os

from .results import valid_candidates


def candidate_label(d):
    label = d["category"] or f"unmapped [{d['raw_phrase']}]"
    return f"{d['id']} {label} | {d['model_score'] * 100:.1f}% model score"


def summary(result):
    retained = valid_candidates(result)
    counts = Counter(d["category"] or "unmapped" for d in retained)
    lines = [f"{result['source']['image_id']}: {result['status']}"]
    lines.extend(candidate_label(d) for d in retained)
    for d in result["detections"]:
        if d["state"] == "invalid":
            reasons = ", ".join(v["reason"] for v in d["numeric_diagnostics"])
            lines.append(f"{d['id']} invalid (excluded): {reasons}")
        elif d["state"] == "duplicate":
            lines.append(f"{d['id']} duplicate of {d['duplicate_of']} (excluded from summary counts)")
    lines.append("Retained candidate predictions: " + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "0"))
    lines.append(f"Object candidates={sum(d['level'] == 'object' for d in retained)}; region candidates={sum(d['level'] == 'region' for d in retained)}; unknown-level candidates={sum(d['level'] == 'unknown' for d in retained)}")
    lines.append("Physical object count: unknown; independent region count: unknown")
    if result["status"] == "no_detections":
        lines.append("No candidates above the configured thresholds; not proof of no trash.")
    if result["error"]:
        lines.append(f"Processing error [{result['error']['stage']}/{result['error']['code']}]: {result['error']['message']}")
    return "\n".join(lines)


def draw_overlays(ax, image, result):
    """Draw on caller-owned axes; pixel edges align to the oriented original."""
    from matplotlib.patches import Rectangle
    width, height = image.size
    ax.imshow(image, extent=(0, width, height, 0), interpolation="nearest")
    colors = {"individual_litter": "lime", "group_of_individual_litter": "cyan", "garbage_pile": "orange", None: "magenta"}
    for d in valid_candidates(result):
        x0, y0, x1, y1 = d["box_xyxy"]
        color = colors[d["category"]]
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor=color, linewidth=2))
        ax.text(x0, max(0, y0), candidate_label(d), color="black", fontsize=8,
                verticalalignment="top", bbox={"facecolor": color, "alpha": 0.8, "pad": 2})
    ax.set(xlim=(0, width), ylim=(height, 0), title=f"BANTAI — {result['status']} — scores are not accuracy")
    ax.set_axis_off()


def show_image(image, result, *, cache_dir):
    # Keep Matplotlib's font/config caches in the explicitly configured cache.
    mpl_cache = Path(cache_dir) / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib
    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt
    figure = None
    try:
        figure, ax = plt.subplots(figsize=(12, 8))
        draw_overlays(ax, image, result)
        figure.canvas.draw()  # Rendering must succeed before normal-return success.
        plt.show(block=True)
    finally:
        if figure is not None:
            plt.close(figure)
