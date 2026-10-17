"""Native in-memory TkAgg viewing and bounded diagnostic terminal tables."""

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


class NativeDisplayError(RuntimeError):
    """Native display failed; saved predictions remain independent."""


def safe_text(value):
    return json.dumps(str(value), ensure_ascii=True)[1:-1].replace("|", r"\u007c")


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
    raw = result.get("query_diagnostics")
    if mode == "full" and raw is not None:
        return raw["queries"]
    by_id = {q["id"]: q for q in raw["queries"]} if raw is not None else {}
    return [by_id.get(d["id"], dict(d, threshold_pass=True)) for d in passing_candidates(result)]


def quoted_text(value):
    """Quote raw text so empty strings and controls remain explicit."""
    return json.dumps(value,ensure_ascii=True).replace("|",r"\u007c")


def box_text(box):
    if box is None:
        return "null"
    return "["+", ".join(f"{v:.2f}" if finite_number(v) else "null" for v in box)+"]"


def detection_fields(result, query, *, full=False):
    """Presentation-only record; never remap phrases or round persisted data."""
    d=next((d for d in result["detections"] if d["id"]==query["id"]),None)
    record=d if d is not None else query
    phrase=record["raw_phrase"]
    category=record["category"]
    unmapped=category is None and phrase is not None
    mapping=record.get("mapping_reason","not recorded")
    invalid=bool(record["numeric_diagnostics"]) or bool(query.get("invalid_token_positions"))
    passed=query.get("threshold_pass",True)
    model_score=query["model_score"] if full else record["model_score"]
    status=decision(model_score,passed,invalid)
    fields=[("Image ID",quoted_text(result["source"]["image_id"])),
            ("Query ID",safe_text(query["id"])),
            ("Raw decoded phrase",quoted_text(phrase) if phrase is not None else "not decoded"),
            ("Mapped category",safe_text(category) if category is not None else "unmapped" if unmapped else "not assigned (not decoded)"),
            ("Mapping reason",safe_text(mapping)),
            ("Model score",score_text(model_score)),
            ("Threshold status",status),
            ("Box xyxy (pixels)",box_text(record.get("box_xyxy",record.get("raw_box_xyxy")))),
            ("Policy state",record.get("state","diagnostic only"))]
    if full and "raw_box_xyxy" in record and record.get("box_xyxy") != record["raw_box_xyxy"]:
        fields.append(("Raw box xyxy",box_text(record["raw_box_xyxy"])))
    return fields


def table_pages(headers, rows, *, width, page_rows):
    """Responsive wrapped tables: preserve complete headers/values without clipping."""
    width = max(24,width)
    status_width = max(len(headers[-1]), max((len(str(r[-1])) for r in rows),default=0))
    status_width = max(15,status_width)
    if len(headers) == 2:
        label_width=min(24,max(10,width//3),max(len(headers[0]),max((len(str(r[0])) for r in rows),default=0)))
        widths=[label_width,width-label_width-3]
        flexible=1
    elif len(headers) == 4:
        query_width = min(8,max(5,max((len(str(r[0])) for r in rows),default=5)))
        widths = [query_width,width-query_width-6-status_width-9,6,status_width]
        flexible = 1
    else:
        widths = [width-6-status_width-6,6,status_width]
        flexible = 0
    if widths[flexible] < 8:
        # Narrow text-only streams: deliberate stacked fields, not terminal wrap.
        page = []
        for row in rows:
            for h,cell in zip(headers,row):
                for line in textwrap.wrap(f"{h}: {cell}",width=width):
                    if len(page) == page_rows:
                        yield page
                        page=[]
                    page.append(line)
        if page or not rows:
            yield page or ["(no rows)"]
        return
    labels = list(headers)
    def render(row):
        cells = [textwrap.wrap(str(c),width=w,break_long_words=True,break_on_hyphens=False) or [""] for c,w in zip(row,widths)]
        return [" | ".join((c[i] if i<len(c) else "").ljust(w) for c,w in zip(cells,widths)).rstrip()
                for i in range(max(map(len,cells)))]
    heading = render(labels)+["-+-".join("-"*w for w in widths)]
    page=[]
    for row in rows:
        lines=render(row)
        if page and len(page)+len(lines)>page_rows:
            yield heading+page
            page=[]
        for line in lines:
            if len(page)==page_rows:
                yield heading+page
                page=[]
            page.append(line)
    if page or not rows:
        yield heading+(page or ["(no rows)"])


def presentation_blocks(result, mode="summary", *, width=80, below_count=0, page_rows=10, include_special=False):
    if mode not in ("summary","full"):
        raise ValueError("terminal mode must be summary or full")
    raw=result.get("query_diagnostics")
    detections=passing_candidates(result)
    queries=selected_queries(result,mode)
    full=mode=="full" and raw is not None
    by_id={d["id"]:d for d in result["detections"]}
    yield ["DETECTION SUMMARY"]+textwrap.wrap("Image ID: "+quoted_text(result["source"]["image_id"]),width=width)+textwrap.wrap("Processing status: "+result["status"],width=width)
    for q in queries:
        fields=detection_fields(result,q,full=full)
        for page in table_pages(("Field","Value"),fields,width=width,page_rows=page_rows):
            yield [f"DETECTION {safe_text(q['id'])}"]+page
    if not detections:
        yield textwrap.wrap("No valid retained queries pass the configured box threshold; no boxes are drawn.",width=width)
    counts=Counter(d["category"] or "unmapped" for d in detections)
    info=["PASS is a strict threshold decision, never correctness.",
          "Retained candidates: "+(", ".join(f"{k}={v}" for k,v in sorted(counts.items())) or "0"),
          "Physical object count: unknown; independent region count: unknown"]
    if full:
        info.append("Full diagnostics include every saved raw query. Only valid retained passing boxes are drawn.")
    elif mode=="full":
        info.append("Raw diagnostics unavailable in this older record; only saved retained predictions can be shown.")
    if any(d["category"] is None for d in detections):
        info.append("An unmapped retained/PASS prediction is not a failed detection. Its decoded phrase has no assigned category; incomplete phrases are never forced into categories.")
    yield [line for text in info for line in textwrap.wrap(text,width=width)]
    yield ["TEXT-TOKEN SCORE DETAILS"]
    token_count=0
    for q in queries if raw is not None else []:
        token_rows=[]
        for t,score,passed in zip(raw["tokens"],q["token_scores"],q["token_threshold_pass"]):
            if t["token"]=="[PAD]" or (t["special"] and not (full and include_special)):
                continue
            invalid=t["position"] in q["invalid_token_positions"] or not finite_number(score) or not 0<=score<=1
            if not full and (not passed or invalid):
                continue
            label=f"{safe_text(t['token'])} [pos={t['position']} id={t['token_id']}]"
            if t["special"]:
                label+=" SPECIAL"
            token_rows.append((label,score_text(score),decision(score,passed,invalid)))
        if token_rows:
            state=by_id[q["id"]]["state"] if q["id"] in by_id else "diagnostic only"
            for page in table_pages(("Token","Score","Text Threshold Status"),token_rows,width=width,page_rows=page_rows):
                yield [f"QUERY {q['id']} ({state})"]+page
        elif full:
            yield [f"QUERY {q['id']}","No ordinary token scores available."]
        token_count+=len(token_rows)
        if full and q["numeric_diagnostics"]:
            yield [f"NUMERIC DIAGNOSTICS {q['id']}"]+textwrap.wrap(safe_text(q["numeric_diagnostics"]),width=width)
    if not token_count:
        message="No ordinary text tokens pass the configured text threshold for the retained queries."
        if raw is None:
            message="Token diagnostics are unavailable in this older result; no token scores were invented."
        yield textwrap.wrap(message,width=width)
    policy="Full mode shows below-threshold/invalid diagnostics as well as passing scores." if full else "Summary shows passing retained queries and passing ordinary tokens only."
    yield textwrap.wrap(policy+" Complete raw precision and all diagnostics remain in the saved JSON.",width=width)
    if result["error"]:
        yield ["PROCESSING ERROR"]+textwrap.wrap(safe_text(result["error"]),width=width)
    inference=result["inference"]
    yield ["CONFIGURATION",f"Box threshold: {inference['threshold']:.4f}",f"Text threshold: {inference['text_threshold']:.4f}",
           "Model: Grounding DINO Tiny","Scores in [0,1]; not accuracy."]


def summary(result, mode=None, **overrides):
    settings = result.get("display", {})
    mode = mode or settings.get("terminal_mode","summary")
    options = dict(width=max(24, terminal_size().columns-1), below_count=0,
                   page_rows=settings.get("page_rows",10), include_special=settings.get("include_special_tokens",False))
    options.update(overrides)
    return "\n\n".join("\n".join(block) for block in presentation_blocks(result, mode, **options))


def native_details(result):
    """Full readable records in the same window, away from image overlays."""
    rows=[]
    for d in passing_candidates(result):
        rows.append(f"{d['id']} | {d['model_score']*100:.1f}% model score\n"
                    f"Category: {d['category'] or 'unmapped'}\n"
                    f"Phrase: {quoted_text(d['raw_phrase'])}\n"
                    f"Box xyxy (original px):\n{box_text(d['box_xyxy'])}")
    return "\n\n".join(rows) if rows else "No valid retained queries pass the box threshold.\nOriginal image displayed without boxes."


def draw_overlays(ax, image, result):
    """Draw the EXIF-oriented source in pixel-edge coordinates, without resizing."""
    from matplotlib.patches import Rectangle
    colors={"garbage_pile":"#ffb000", "individual_litter":"#00ff70",
            "group_of_individual_litter":"#00dfff", None:"#ff66ff"}
    width,height=image.size
    ax.imshow(image,origin="upper",extent=(0,width,height,0),interpolation="antialiased")
    for d in passing_candidates(result):
        x0,y0,x1,y1=d["box_xyxy"]
        color=colors[d["category"]]
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,
                               edgecolor=color,linewidth=2,zorder=3))
        label=d["category"] or "unmapped"
        bottom=y0>height*.8
        right=x0>width*.6
        ax.annotate(label,(x1 if right else x0,y1 if bottom else y0),xytext=(-4 if right else 4,4 if bottom else -4),
                    textcoords="offset points",va="bottom" if bottom else "top",ha="right" if right else "left",
                    fontsize=10,color="white",clip_on=True,zorder=4,
                    bbox=dict(boxstyle="round,pad=0.3",facecolor="black",edgecolor=color,alpha=.9))
    ax.set(xlim=(0,width),ylim=(height,0),xlabel="x (original pixels)",ylabel="y (original pixels)")
    ax.set_aspect("equal",adjustable="box")
    ax.set_title("Model scores are not accuracy. Zoom / Pan: toolbar; Home: reset. Close to continue.",fontsize=10)


def open_native_window(figure, title, *, details=None):
    """Block until normal window closure; propagate Tk callback/render failures."""
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    class ViewToolbar(NavigationToolbar2Tk):
        # Viewer-only navigation: no image save/export or subplot pop-ups.
        toolitems=tuple(item for item in NavigationToolbar2Tk.toolitems
                        if item[0] in ("Home","Back","Forward","Pan","Zoom"))

    root=None
    try:
        root=tk.Tk()
        root.title(title)
        root.geometry(f"{min(1400,int(root.winfo_screenwidth()*.9))}x{min(950,int(root.winfo_screenheight()*.85))}")
        closed=False
        failures=[]
        def close():
            nonlocal closed
            closed=True
            root.quit()
        def callback_error(kind,error,traceback):
            failures.append(error)
            root.quit()
        root.report_callback_exception=callback_error
        root.protocol("WM_DELETE_WINDOW",close)
        image_frame=root
        if details is not None:
            from tkinter.scrolledtext import ScrolledText
            panel=ScrolledText(root,width=48,wrap=tk.WORD,font=("Consolas",11),
                               background="#171717",foreground="white",padx=12,pady=12)
            panel.insert("1.0","DETECTION DETAILS\nScores are model outputs, not accuracy.\n\n"+details)
            panel.configure(state=tk.DISABLED)
            panel.pack(side=tk.RIGHT,fill=tk.Y)
            image_frame=tk.Frame(root)
            image_frame.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
        canvas=FigureCanvasTkAgg(figure,master=image_frame)
        toolbar=ViewToolbar(canvas,image_frame,pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM,fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=True)
        canvas.draw()
        root.mainloop()
        if failures:
            raise NativeDisplayError(f"Native viewer callback failed ({type(failures[0]).__name__})") from failures[0]
        if not closed:
            raise NativeDisplayError("Native viewer ended without normal window closure")
    finally:
        if root is not None:
            # Figure.clear() updates its toolbar. Detach it before Tk widgets die.
            figure.canvas.toolbar=None
            root.destroy()


def show_image(image,result,*,cache_dir=None,stream=None,input_stream=None):
    """Use existing TkAgg dependencies; stdout need not be an interactive terminal."""
    from matplotlib.figure import Figure
    stream=sys.stdout if stream is None else stream
    stream.write(f"IMAGE: {safe_text(Path(result['source']['image_id']).name)}\n")
    stream.write(summary(result,mode="summary")+"\n")
    stream.write("Opening native TkAgg viewer; close its window to continue.\n")
    stream.flush()
    figure=Figure(figsize=(12,8),dpi=100,layout="constrained")
    try:
        draw_overlays(figure.subplots(),image,result)
        open_native_window(figure,"BANTAI | "+safe_text(result["source"]["image_id"]),details=native_details(result))
    except NativeDisplayError:
        raise
    except Exception as exc:
        raise NativeDisplayError(f"Native TkAgg viewer failed ({type(exc).__name__}): {exc}") from exc
    finally:
        figure.clear()


def show_synthetic(*,stream=None,preview_width=120):
    """Independent native-window check, without model execution or file writes."""
    from PIL import Image,ImageDraw
    from matplotlib.figure import Figure
    stream=sys.stdout if stream is None else stream
    stream.write("SYNTHETIC NATIVE VIEWER CHECK: fixture boxes/IDs only; no model inference.\n")
    stream.flush()
    result={"inference":{"threshold":.35},"detections":[
        {"id":"test0","state":"retained","model_score":.86,"numeric_diagnostics":[],
         "raw_phrase":"synthetic blue circle","category":"individual_litter","box_xyxy":[40,35,145,145]},
        {"id":"test1","state":"retained","model_score":.91,"numeric_diagnostics":[],
         "raw_phrase":"synthetic yellow rectangle","category":"garbage_pile","box_xyxy":[175,55,285,155]}]}
    with Image.new("RGB",(320,180),"#ddddee") as image:
        draw=ImageDraw.Draw(image)
        draw.ellipse((50,45,135,135),fill="blue")
        draw.rectangle((185,65,275,145),fill="yellow")
        figure=Figure(figsize=(12,8),dpi=100,layout="constrained")
        try:
            draw_overlays(figure.subplots(),image,result)
            open_native_window(figure,"BANTAI | synthetic viewer check",details=native_details(result))
        finally:
            figure.clear()
    stream.write("Native window closed normally. Visual appearance remains unverified unless observed.\n")
    stream.flush()
