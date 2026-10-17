"""Interactive ANSI/Unicode previews and bounded tables, with honest capability errors."""

import json
import os
import shutil
import struct
import sys
import textwrap
from collections import Counter
from pathlib import Path

from .configuration import finite_number
from .results import valid_candidates


class TerminalDisplayError(RuntimeError):
    """The current output cannot display an inline ANSI/Unicode image."""


def safe_text(value):
    return json.dumps(str(value), ensure_ascii=True)[1:-1]


def score_text(value):
    return f"{value:.4f}" if finite_number(value) and 0 <= value <= 1 else "null"


def candidate_label(d):
    return f"{d['id']} {d['category'] or 'unmapped'} | {score_text(d['model_score'])} model score"


def decision(value, passed, invalid=False):
    if invalid or not finite_number(value) or not 0 <= value <= 1:
        return "INVALID"
    return "PASS" if passed else "BELOW THRESHOLD"


def terminal_size(stream=None):
    try:
        return os.get_terminal_size((stream or sys.stdout).fileno())
    except (AttributeError, OSError, ValueError):
        return shutil.get_terminal_size(fallback=(80, 24))


def terminal_facts(stream=None, input_stream=None):
    stream = sys.stdout if stream is None else stream
    input_stream = sys.stdin if input_stream is None else input_stream
    size = terminal_size(stream)
    return {"stdout_isatty": bool(stream.isatty()), "stdin_isatty": bool(input_stream.isatty()),
            "encoding": getattr(stream,"encoding",None), "columns": size.columns, "rows": size.lines,
            "TERM": os.environ.get("TERM",""), "COLORTERM": os.environ.get("COLORTERM",""),
            "TERM_PROGRAM": os.environ.get("TERM_PROGRAM",""), "windows_terminal": bool(os.environ.get("WT_SESSION"))}


def enable_windows_vt(stream):
    if os.name != "nt":
        return False
    import ctypes
    import msvcrt
    try:
        handle = ctypes.c_void_p(msvcrt.get_osfhandle(stream.fileno()))
    except (AttributeError, OSError, ValueError):
        return False
    mode = ctypes.c_ulong()
    kernel = ctypes.windll.kernel32
    if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
        return False  # ConPTY may advertise capability through its environment.
    if not kernel.SetConsoleMode(handle, mode.value | 4):
        raise TerminalDisplayError("Inline image unavailable: Windows virtual-terminal processing could not be enabled.")
    return True


def terminal_capabilities(stream):
    size = terminal_size(stream)
    facts = json.dumps(terminal_facts(stream),ensure_ascii=True)
    if not stream.isatty():
        raise TerminalDisplayError(f"Inline image unavailable: stdout is noninteractive/redirected ({facts}). Run in the VS Code integrated terminal; use --no-display for text/JSON only.")
    term = os.environ.get("TERM", "")
    color = os.environ.get("COLORTERM", "").lower()
    known = os.environ.get("TERM_PROGRAM", "").lower() in ("vscode", "wezterm", "iterm.app") or bool(os.environ.get("WT_SESSION"))
    # Windows consoles often leave TERM unset or inherited as dumb. A successful
    # native VT enablement or a known terminal is stronger evidence than TERM.
    native_vt = enable_windows_vt(stream)
    if not (native_vt or known or color in ("truecolor", "24bit") or "direct" in term):
        raise TerminalDisplayError(f"Inline image unavailable: ANSI truecolor support is unconfirmed ({facts}). Use a truecolor VS Code terminal.")
    try:
        "\u2580".encode(stream.encoding or "utf-8")
    except (LookupError, UnicodeError):
        raise TerminalDisplayError(f"Inline image unavailable: output encoding cannot represent Unicode half blocks ({facts}). Use UTF-8.") from None
    if size.columns < 50 or size.lines < 12:
        raise TerminalDisplayError(f"Inline image unavailable: terminal too small ({facts}); resize to at least 50 columns x 12 rows.")
    return size


def float32_pass(score, threshold):
    """Match the pinned float32 tensor comparison without importing torch."""
    if not finite_number(score) or not 0 <= score <= 1:
        return False
    f32 = lambda value: struct.unpack("f", struct.pack("f", value))[0]
    return f32(score) > f32(threshold)


def passing_candidates(result):
    raw = result.get("query_diagnostics")
    by_id = {q["id"]: q for q in raw["queries"]} if raw is not None else None
    passing = []
    for d in valid_candidates(result):
        if d["numeric_diagnostics"] or not finite_number(d["model_score"]):
            continue
        if by_id is not None:
            q = by_id.get(d["id"])
            if (q is None or not q["threshold_pass"] or q["numeric_diagnostics"]
                    or q.get("invalid_token_positions") or not finite_number(q["model_score"])):
                continue
        elif not float32_pass(d["model_score"], result["inference"]["threshold"]):
            continue
        passing.append(d)
    return passing


def selected_queries(result, mode="summary", below_count=0):
    # Legacy mode/selection arguments cannot enable excluded console rows.
    raw = result.get("query_diagnostics")
    by_id = {q["id"]: q for q in raw["queries"]} if raw is not None else {}
    return [by_id.get(d["id"], dict(d, threshold_pass=True)) for d in passing_candidates(result)]


def table_pages(headers, rows, *, width, page_rows):
    """Wrap cells deliberately inside bounded columns; never rely on terminal wrap."""
    width = max(24, width)
    query_width = min(8, max(5, max((len(str(r[0])) for r in rows), default=5)))
    widths = [query_width, width-query_width-6-15-9, 6, 15]
    if widths[1] < 5:
        # Very narrow text-only output: stack each row with its field names.
        for i in range(0, max(1, len(rows)), page_rows):
            lines = []
            for row in rows[i:i+page_rows]:
                for h, cell in zip(headers, row):
                    lines.extend(textwrap.wrap(f"{h}: {cell}", width=width) or [""])
                lines.append("")
            yield lines or ["(no rows)"]
        return
    def render(row):
        cells = [textwrap.wrap(str(c), width=w, break_long_words=True, break_on_hyphens=False) or [""] for c,w in zip(row,widths)]
        return [" | ".join((c[i] if i<len(c) else "").ljust(w) for c,w in zip(cells,widths)).rstrip()
                for i in range(max(map(len,cells)))]
    heading = render(headers)
    divider = "-+-".join("-"*w for w in widths)
    # Bound physical lines as well as logical rows, so long subwords/phrases do
    # not silently grow one page beyond terminal height.
    page = []
    for row in rows:
        for line in render(row):
            if len(page) == page_rows:
                yield heading + [divider] + page
                page = []
            page.append(line)
    if page or not rows:
        yield heading + [divider] + (page or ["(no rows)"])


def presentation_blocks(result, mode="summary", *, width=80, below_count=0, page_rows=10, include_special=False):
    if mode not in ("summary", "full"):
        raise ValueError("terminal mode must be summary or full")
    # Both historical console modes now obey the owner's passing-only policy.
    raw = result.get("query_diagnostics")
    detections = passing_candidates(result)
    queries = selected_queries(result)
    query_rows = [(d["id"], safe_text(d["raw_phrase"]), score_text(d["model_score"]), "PASS") for d in detections]
    for page in table_pages(("Query", "Detected Phrase", "Score", "Box Status"), query_rows, width=width, page_rows=page_rows):
        yield ["DETECTION SUMMARY"] + page
    if not detections:
        yield textwrap.wrap("No valid retained queries pass the configured box threshold.", width=width)
    counts = Counter(d["category"] or "unmapped" for d in detections)
    info = ["PASS is a strict threshold decision, never correctness.",
            "Retained candidates: " + (", ".join(f"{k}={v}" for k,v in sorted(counts.items())) or "0"),
            "Physical object count: unknown; independent region count: unknown"]
    yield [line for text in info for line in textwrap.wrap(text, width=width)]
    for d in detections:
        yield [line for text in [f"BOX {d['id']}: {d['box_xyxy']}", f"CATEGORY {d['id']}: {d['category'] or 'unmapped'}"] for line in textwrap.wrap(text,width=width)]
    token_rows = []
    if raw is not None:
        for q in queries:
            for t, score, passed in zip(raw["tokens"], q["token_scores"], q["token_threshold_pass"]):
                if (t["special"] or t["token"] == "[PAD]" or not passed or not finite_number(score)
                        or not 0 <= score <= 1 or t["position"] in q["invalid_token_positions"]):
                    continue
                label = f"{safe_text(t['token'])} [pos={t['position']} id={t['token_id']}]"
                token_rows.append((q["id"], label, score_text(score), "PASS"))
    for page in table_pages(("Query", "Token", "Score", "Text Threshold Status"), token_rows, width=width, page_rows=page_rows):
        yield ["TEXT-TOKEN SCORE DETAILS"] + page
    if not token_rows:
        message = "No ordinary text tokens pass the configured text threshold for the retained queries."
        if raw is None:
            message = "Token diagnostics are unavailable in this older result; no token scores were invented."
        yield textwrap.wrap(message,width=width)
    yield textwrap.wrap("Console shows passing retained queries and passing ordinary tokens only. Complete raw scores and excluded/invalid diagnostics remain in the saved JSON.",width=width)
    if result["error"]:
        yield ["PROCESSING ERROR"] + textwrap.wrap(safe_text(result["error"]), width=width)
    inference = result["inference"]
    yield ["CONFIGURATION", f"Box threshold: {inference['threshold']:.4f}", f"Text threshold: {inference['text_threshold']:.4f}",
           "Model: Grounding DINO Tiny", "Scores in [0,1]; not accuracy."]


def summary(result, mode="summary", **overrides):
    settings = result.get("display", {})
    options = dict(width=max(24, terminal_size().columns-1), below_count=0,
                   page_rows=settings.get("page_rows",10), include_special=False)
    options.update(overrides)
    return "\n\n".join("\n".join(block) for block in presentation_blocks(result, mode, **options))


def terminal_image_lines(image, result, *, columns=None, rows=None):
    """Square-pixel RGB half blocks with query IDs at full terminal-cell size."""
    from PIL import Image, ImageDraw
    size = terminal_size()
    columns = max(1, min(240, columns if columns is not None else size.columns-1))
    rows = max(1, min(120, rows if rows is not None else size.lines-5))
    scale = min(columns/image.width, (2*rows)/image.height)
    width, height = max(1,int(image.width*scale)), max(1,int(image.height*scale))
    colors = {"garbage_pile": (255,165,0), "individual_litter": (0,255,0), "group_of_individual_litter": (0,255,255), None: (255,0,255)}
    labels = {}
    with image.resize((width,height), Image.Resampling.LANCZOS).convert("RGB") as preview:
        draw = ImageDraw.Draw(preview)
        for d in passing_candidates(result):
            x0,y0,x1,y1 = d["box_xyxy"]
            box = (min(width-1,x0*width/image.width),min(height-1,y0*height/image.height),
                   min(width-1,x1*width/image.width),min(height-1,y1*height/image.height))
            color = colors[d["category"]]
            draw.rectangle(box,outline=color,width=1)
            ident = d["id"]
            lx = max(0,min(int(box[0]),width-len(ident)))
            preferred = min((height-1)//2,int(box[1])//2)
            candidates = sorted(range((height+1)//2), key=lambda y: abs(y-preferred))
            for ly in candidates:
                if all((lx+j,ly) not in labels for j in range(len(ident))):
                    for j,char in enumerate(ident):
                        labels[lx+j,ly] = (char,color)
                    break
        pixels = preview.load()
        for y in range(0,height,2):
            line = []
            for x in range(width):
                if (x,y//2) in labels:
                    char,(r,g,b) = labels[x,y//2]
                    line.append(f"\x1b[38;2;0;0;0m\x1b[48;2;{r};{g};{b}m{char}")
                else:
                    r,g,b = pixels[x,y]
                    br,bg,bb = pixels[x,min(y+1,height-1)]
                    line.append(f"\x1b[38;2;{r};{g};{b}m\x1b[48;2;{br};{bg};{bb}m\u2580")
            yield "".join(line)+"\x1b[0m"


def show_image(image, result, *, cache_dir=None, stream=None, input_stream=None):
    stream = sys.stdout if stream is None else stream
    input_stream = sys.stdin if input_stream is None else input_stream
    # No escape bytes are emitted until capabilities have been checked.
    stream.write(f"IMAGE: {safe_text(Path(result['source']['image_id']).name)}\n")
    stream.flush()
    size = terminal_capabilities(stream)
    if not input_stream.isatty():
        raise TerminalDisplayError("Inline display requires interactive stdin for paging; run in the VS Code terminal, or use --no-display for text/JSON.")
    width = size.columns-1
    settings = result.get("display",{})
    mode = settings.get("terminal_mode","summary")
    def write(lines):
        stream.write("\n".join(lines)+"\n")
        stream.flush()
    def pause():
        write(textwrap.wrap("Enter: continue; Ctrl+C: stop (JSON already saved).", width=width))
        if input_stream.readline() == "":
            raise TerminalDisplayError("Terminal input closed during paging; display unfinished.")
    try:
        for line in terminal_image_lines(image,result,columns=width,rows=size.lines-5):
            write([line])
        # Preserve a chance to inspect the actual image before tables scroll it.
        pause()
        blocks = presentation_blocks(result,mode,width=width,
                     below_count=0,
                     page_rows=min(settings.get("page_rows",10),max(1,size.lines-10)),
                     include_special=False)
        used = 0
        for block in blocks:
            budget = size.lines-4
            if used and used+len(block)+1 > budget:
                pause()
                used = 0
            for line in [""]+block:
                if used == budget:
                    pause()
                    used = 0
                write([line])
                used += 1
        # Let the last page be inspected before the next image can scroll it.
        pause()
    finally:
        stream.write("\x1b[0m\n")
        stream.flush()


def show_synthetic(*, stream=None):
    """Independent renderer exercise: all pixels/boxes are software fixtures."""
    from PIL import Image, ImageDraw
    stream = sys.stdout if stream is None else stream
    stream.write("SYNTHETIC RENDERER CHECK: fixture boxes/IDs only; no model inference.\n")
    stream.write(json.dumps(terminal_facts(stream),ensure_ascii=True)+"\n")
    stream.write("IMAGE: synthetic-renderer-check\n")
    stream.flush()
    size = terminal_capabilities(stream)
    fixture = {"inference": {"threshold": .35}, "detections": [
        {"id": "test0", "state": "retained", "model_score": 1.0, "numeric_diagnostics": [],
         "box_xyxy": [40,35,145,145], "category": "individual_litter"},
        {"id": "test1", "state": "retained", "model_score": 1.0, "numeric_diagnostics": [],
         "box_xyxy": [175,55,285,155], "category": "garbage_pile"}]}
    with Image.new("RGB",(320,180)) as image:
        image.putdata([(x*255//319,y*255//179,120) for y in range(180) for x in range(320)])
        draw = ImageDraw.Draw(image)
        draw.ellipse((50,45,135,135),fill=(40,70,220))
        draw.rectangle((185,65,275,145),fill=(240,220,60))
        try:
            for line in terminal_image_lines(image,fixture,columns=size.columns-1,rows=size.lines-8):
                stream.write(line+"\n")
            stream.flush()
        finally:
            stream.write("\x1b[0m\n")
            stream.flush()
    stream.write("Renderer output returned. Visual verification remains pending until the image, boxes and test0/test1 IDs are actually inspected.\n")
    stream.flush()
