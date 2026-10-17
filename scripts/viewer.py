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


def draw_overlays(ax,image,result):
    from .gui_viewer import draw_overlays as draw
    return draw(ax,image,result)


class DisplayBatch:
    """Own in-memory image copies until one batch window closes."""
    def __init__(self):
        self.entries=[]

    def add(self,image,result):
        self.entries.append((image.copy(),result))

    def show(self):
        if self.entries:
            from .gui_viewer import open_results_window
            try:
                open_results_window(self.entries)
            except NativeDisplayError:
                raise
            except Exception as exc:
                raise NativeDisplayError(f"Scrollable TkAgg viewer failed ({type(exc).__name__}): {exc}") from exc

    def close(self):
        for image,_ in self.entries:image.close()
        self.entries.clear()


def show_image(image,result,*,batch=None,cache_dir=None,stream=None,input_stream=None):
    """Queue for a shared tab, or show a single-image tab for direct callers."""
    if batch is not None:
        batch.add(image,result)
        return
    session=DisplayBatch()
    try:
        session.add(image,result)
        session.show()
    finally:
        session.close()


def show_synthetic(*,stream=None,preview_width=120):
    """Three software-only images exercise the complete scrollable tab."""
    from PIL import Image,ImageDraw
    from .configuration import load_config
    from .results import candidate,make_result
    stream=sys.stdout if stream is None else stream
    stream.write("SYNTHETIC SCROLLABLE TAB: software fixtures only; no model inference.\n")
    stream.flush()
    config=load_config();config.update(threshold=.35,text_threshold=.25)
    batch=DisplayBatch()
    try:
        for index in range(3):
            with Image.new("RGB",(640,400),"#ddddee") as image:
                draw=ImageDraw.Draw(image);draw.rectangle((80,100,400,300),fill="blue")
                ds=[] if index==2 else [candidate("q0","a group" if index==0 else "discarded item with a long incomplete phrase",[80,100,400,300],.86,640,400),
                    candidate("q1","an individual",[620,0,640,80],.91,640,400)]
                source=dict(image_id=f"synthetic-{index+1}",sha256="a"*64,width=640,height=400,coordinate_frame="exif_oriented_original")
                result=make_result("synthetic",source,config,detections=ds,inference_completed=True,library_version="4.51.3",
                    execution={"actual_device":"cpu","environment_manifest":"synthetic-only","preprocessing_manifest":"synthetic-only"})
                result["query_diagnostics"]={"tokens":[{"token":"group","position":1,"token_id":123,"special":False}],
                    "queries":[{"id":d["id"],"threshold_pass":True,"numeric_diagnostics":[],"invalid_token_positions":[],
                                "model_score":d["model_score"],"token_scores":[.8],"token_threshold_pass":[True]} for d in ds]}
                batch.add(image,result)
        batch.show()
    finally:
        batch.close()
    stream.write("Scrollable window closed normally. Visual verification requires observation.\n")
    stream.flush()
