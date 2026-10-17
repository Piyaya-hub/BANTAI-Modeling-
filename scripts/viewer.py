"""ANSI truecolor/Unicode terminal viewing; no GUI, image files or exports."""

import json
import shutil
import sys
from collections import Counter

from .results import valid_candidates


def safe_text(value):
    # Escape controls from filenames/model text so they cannot inject terminal commands.
    return json.dumps(str(value), ensure_ascii=True)[1:-1]


def percent(value):
    return "null" if value is None else f"{value * 100:.6f}%"


def candidate_label(d):
    return f"{d['id']} {d['category'] or 'unmapped'} | {percent(d['model_score'])} model score"


def summary(result, mode="summary"):
    if mode not in ("summary", "full"):
        raise ValueError("terminal mode must be summary or full")
    lines = [f"{safe_text(result['source']['image_id'])}: {result['status']}",
             "Detection results (PASS = retained valid prediction; FAIL = excluded; not correctness)",
             "ID | PHRASE / CATEGORY | BOX xyxy (oriented pixels) | MODEL SCORE | STATUS"]
    for d in result["detections"]:
        status = "PASS" if d["state"] == "retained" else "FAIL"
        lines.append(f"{safe_text(d['id'])} | {safe_text(d['raw_phrase'])} / {d['category'] or 'unmapped'} | "
                     f"{d['box_xyxy']} | {percent(d['model_score'])} | {status} ({d['state']})")
        if d["raw_box_xyxy"] != d["box_xyxy"]:
            lines.append(f"  raw box: {d['raw_box_xyxy']}")
        if d["numeric_diagnostics"]:
            lines.append(f"  numeric diagnostics: {safe_text(d['numeric_diagnostics'])}")
    if not result["detections"]:
        lines.append("No extracted detections; see processing status and raw diagnostics.")
    retained = valid_candidates(result)
    counts = Counter(d["category"] or "unmapped" for d in retained)
    lines.append("Retained candidate predictions: " + (", ".join(f"{k}={v}" for k,v in sorted(counts.items())) or "0"))
    lines.append(f"Object candidates={sum(d['level'] == 'object' for d in retained)}; region candidates={sum(d['level'] == 'region' for d in retained)}; unknown-level candidates={sum(d['level'] == 'unknown' for d in retained)}")
    if result["status"] == "no_detections":
        lines.append("No candidates above configured thresholds; not proof of no trash.")
    lines.append("Physical object count: unknown; independent region count: unknown")
    lines.append("Scores are model scores, not accuracy or calibrated correctness probabilities.")
    if result["error"]:
        lines.append(f"Processing error: {safe_text(result['error'])}")
    diagnostics = result.get("query_diagnostics")
    if diagnostics is not None:
        queries, tokens = diagnostics["queries"], diagnostics["tokens"]
        lines.append(f"Raw diagnostics: {len(queries)} queries x {len(tokens)} non-padding tokens; complete scores saved in JSON.")
        if mode == "full":
            lines.append("Raw PASS/FAIL = strict score > threshold only; invalid tokens FAIL. Special tokens are identified, not phrase labels.")
            for q in queries:
                status = "PASS" if q["threshold_pass"] else "FAIL"
                phrase = "not decoded" if q["raw_phrase"] is None else safe_text(q["raw_phrase"])
                lines.append(f"{q['id']} | phrase={phrase} / category={q['category'] or 'unmapped'} | box={q['raw_box_xyxy']} | "
                             f"score={percent(q['model_score'])} | {status} (> {diagnostics['threshold']})")
                if q["numeric_diagnostics"]:
                    lines.append(f"  numeric diagnostics: {safe_text(q['numeric_diagnostics'])}")
                for token, score, passed in zip(tokens, q["token_scores"], q["token_threshold_pass"]):
                    lines.append(f"  token[{token['position']}] id={token['token_id']} {safe_text(token['token'])} "
                                 f"special={token['special']} | {percent(score)} | {'PASS' if passed else 'FAIL'} "
                                 f"(> {diagnostics['text_threshold']})")
    elif mode == "full":
        lines.append("Raw diagnostics unavailable; no scores fabricated.")
    return "\n".join(lines)


def terminal_image_lines(image, result, *, columns=None, rows=None):
    """Resize only an in-memory RGB preview; draw retained boxes and query IDs."""
    from PIL import Image, ImageDraw
    size = shutil.get_terminal_size(fallback=(100, 40))
    columns = max(1, min(240, columns if columns is not None else size.columns - 1))
    rows = max(1, min(120, rows if rows is not None else size.lines - 10))
    scale = min(columns / image.width, (2 * rows) / image.height)
    width, height = max(1, int(image.width * scale)), max(1, int(image.height * scale))
    colors = {"garbage_pile": "orange", "individual_litter": "lime", "group_of_individual_litter": "cyan", None: "magenta"}
    with image.resize((width, height), Image.Resampling.LANCZOS).convert("RGB") as preview:
        draw = ImageDraw.Draw(preview)
        for d in valid_candidates(result):
            x0, y0, x1, y1 = d["box_xyxy"]
            box = (min(width-1, x0*width/image.width), min(height-1, y0*height/image.height),
                   min(width-1, x1*width/image.width), min(height-1, y1*height/image.height))
            color = colors[d["category"]]
            draw.rectangle(box, outline=color, width=1)
            # Query IDs link to the complete labels and scores immediately below.
            draw.text((max(0, min(box[0], width-18)), max(0, box[1]-9)), d["id"], fill=color)
        pixels = preview.load()
        for y in range(0, height, 2):
            line = []
            for x in range(width):
                r,g,b = pixels[x,y]
                br,bg,bb = pixels[x,min(y+1,height-1)]
                line.append(f"\x1b[38;2;{r};{g};{b}m\x1b[48;2;{br};{bg};{bb}m\u2580")
            yield "".join(line) + "\x1b[0m"


def show_image(image, result, *, cache_dir=None, stream=None):
    # cache_dir retained for caller compatibility; no files are written.
    stream = sys.stdout if stream is None else stream
    try:
        for line in terminal_image_lines(image, result):
            stream.write(line + "\n")
        stream.write(summary(result, mode=result.get("display", {}).get("terminal_mode", "summary")) + "\n")
        stream.flush()
    finally:
        # Reset terminal color even if rendering or writing was interrupted.
        stream.write("\x1b[0m")
        stream.flush()
