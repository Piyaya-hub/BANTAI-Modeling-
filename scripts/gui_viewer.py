"""One local scrollable Tk tab. Images and Matplotlib overlays stay in memory."""

import math

from .viewer import passing_candidates, score_text, box_text, quoted_text, finite_number


SECTION_ORDER=("Image", "Detection Summary", "Text-Token Details", "Configuration")


def section_data(result):
    """Passing-only views of saved data; never decode, map or mutate predictions."""
    detections=passing_candidates(result)
    summary=[(d["id"],quoted_text(d["raw_phrase"]),d["category"] or "unmapped",
              d["mapping_reason"],score_text(d["model_score"]),box_text(d["box_xyxy"]),"PASS") for d in detections]
    tokens=[]
    raw=result.get("query_diagnostics")
    if raw is not None:
        ids={d["id"] for d in detections}
        for q in raw["queries"]:
            if q["id"] not in ids:continue
            for t,score,passed in zip(raw["tokens"],q["token_scores"],q["token_threshold_pass"]):
                if (not t["special"] and t["token"]!="[PAD]" and passed
                        and t["position"] not in q["invalid_token_positions"]
                        and finite_number(score) and 0<=score<=1):
                    tokens.append((q["id"],quoted_text(t["token"]),str(t["position"]),str(t["token_id"]),score_text(score),"PASS"))
    inference=result["inference"];model=result["model"]
    config=[("Image ID",result["source"]["image_id"]),
            ("Prompt",inference.get("prompt","not recorded")),
            ("Box threshold",str(inference["threshold"])),("Text threshold",str(inference["text_threshold"])),
            ("Model",model["id"]),("Revision",model["revision"]),
            ("Device used",inference.get("actual_device") or "not recorded"),
            ("Score and threshold units","decimal [0,1]; strict > comparison; not accuracy"),
            ("Label font (points)",str(result.get("display",{}).get("label_font_size",8.5))),
            ("Minimum label font (points)",str(result.get("display",{}).get("label_min_font_size",8))),
            ("Coordinate frame",result["source"]["coordinate_frame"])]
    return {"summary":summary,"tokens":tokens,"configuration":config,
            "token_message":"Token diagnostics unavailable in this saved result." if raw is None else "No ordinary text tokens pass the text threshold for retained queries."}


def covers_border(bounds,ax,detections):
    """Do not obscure any retained box outline with a label background."""
    left,bottom,right,top=bounds
    for detection in detections:
        x0,y0,x1,y1=detection["box_xyxy"]
        xa,ya=ax.transData.transform((x0,y0));xb,yb=ax.transData.transform((x1,y1))
        lo,hi=sorted((ya,yb))
        if (right>=xa-2 and left<=xb+2 and (bottom-2<=ya<=top+2 or bottom-2<=yb<=top+2)
                or top>=lo-2 and bottom<=hi+2 and (left-2<=xa<=right+2 or left-2<=xb<=right+2)):
            return True
    return False


def wrap_pixels(text,width,measure,*,break_words=True):
    """Wrap using actual glyph widths, preserving every decoded character."""
    lines=[]
    for paragraph in text.split("\n"):
        if not paragraph:lines.append("");continue
        while paragraph:
            count=0
            for end in range(1,len(paragraph)+1):
                if measure(paragraph[:end])>width:break
                count=end
            if count==0:return None
            if count<len(paragraph):
                space=paragraph.rfind(" ",0,count)
                if space>0:count=space+1
                elif not break_words:return None
            lines.append(paragraph[:count]);paragraph=paragraph[count:]
    return lines


def phrase_artist(ax,d,detections,settings=None):
    """Lay out against the actual renderer on every draw, including zoom/pan."""
    from matplotlib.artist import Artist
    from matplotlib.text import Text
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import IdentityTransform

    class PhraseLabel(Artist):
        def __init__(self):
            super().__init__()
            self.last_layout=None
            self.query_id=d["id"]
            self.raw_phrase=d["raw_phrase"]
        def draw(self,renderer):
            self.last_layout=None
            x0,y0,x1,_=d["box_xyxy"]
            left,top=ax.transData.transform((x0,y0))
            right,_=ax.transData.transform((x1,y0))
            left=max(left,ax.bbox.x0+1);right=min(right,ax.bbox.x1-1)
            phrase=d["raw_phrase"]
            phrase=phrase if phrase else '""'  # Explicit empty decoded string.
            for break_words in (False,True):
                for size in ((settings or {}).get("label_font_size",8.5),(settings or {}).get("label_min_font_size",8)):
                    font=FontProperties(size=size)
                    pad=size*renderer.dpi/72*.2
                    available=right-left-2*pad-2
                    if available<=0:continue
                    measure=lambda value:renderer.get_text_width_height_descent(value,font,False)[0]
                    if available<measure("MMM"):continue
                    # Prefer matching IDs, but never displace the actual raw phrase.
                    phrase_lines=wrap_pixels(phrase,available,measure,break_words=break_words)
                    for prefix in (d["id"]+" ",""):
                        lines=wrap_pixels(prefix+phrase,available,measure,break_words=break_words)
                        if lines is None or len(lines)>8:continue
                        if prefix and phrase_lines is not None and len(lines)>len(phrase_lines):continue
                        label=Text(left+pad+1,top+5+pad,"\n".join(lines),fontproperties=font,
                                   color="white",va="bottom",ha="left",transform=IdentityTransform(),
                                   parse_math=False,bbox=dict(boxstyle="square,pad=0.2",facecolor="black",edgecolor="none",alpha=.9))
                        label.set_figure(ax.figure)
                        bounds=label.get_window_extent(renderer)
                        if bounds.y1+pad>ax.bbox.y1-1 or bounds.y0-pad<ax.bbox.y0:continue
                        rectangle=(bounds.x0-pad,bounds.y0-pad,bounds.x1+pad,bounds.y1+pad)
                        if covers_border(rectangle,ax,detections):continue
                        preceding=list(ax.artists)[:list(ax.artists).index(self)]
                        if any(a.last_layout and not (rectangle[2]<a.last_layout["bounds"][0]
                               or rectangle[0]>a.last_layout["bounds"][2] or rectangle[3]<a.last_layout["bounds"][1]
                               or rectangle[1]>a.last_layout["bounds"][3]) for a in preceding if hasattr(a,"last_layout")):continue
                        label.draw(renderer)
                        self.last_layout={"bounds":rectangle,
                                          "box_span":(left,right),"box_top":top,"font_size":size,"text":label.get_text()}
                        break
                    if self.last_layout:break  # Otherwise the complete phrase remains in the table.
                if self.last_layout:break
            self.stale=False
    artist=PhraseLabel();artist.set_zorder(5);ax.add_artist(artist)
    return artist


def draw_overlays(ax,image,result):
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator,FuncFormatter
    width,height=image.size
    ax.imshow(image,origin="upper",extent=(0,width,height,0),interpolation="antialiased")
    colors={"garbage_pile":"#ffb000","individual_litter":"#00ff70",
            "group_of_individual_litter":"#00dfff",None:"#ff66ff"}
    detections=passing_candidates(result)
    for d in detections:
        x0,y0,x1,y1=d["box_xyxy"];color=colors[d["category"]]
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=color,linewidth=2,zorder=3))
        phrase_artist(ax,d,detections,result.get("display"))
    # Negative y margin is display space for labels above top-edge boxes;
    # image extent and every box stay in original EXIF-oriented coordinates.
    ax.set(xlim=(0,width),ylim=(height,-height*.22),xlabel="x (original pixels)",ylabel="y (original pixels)")
    ax.set_aspect("equal",adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7,integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7,integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f"{value:g}" if 0<=value<=width else ""))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f"{value:g}" if 0<=value<=height else ""))
    ax.tick_params(top=True,right=True,labeltop=True,labelright=True)


def table_text(headers,rows):
    widths=[max(len(str(h)),max((len(str(row[i])) for row in rows),default=0)) for i,h in enumerate(headers)]
    render=lambda row:" | ".join(str(value).ljust(width) for value,width in zip(row,widths)).rstrip()
    return "\n".join([render(headers),"-+-".join("-"*w for w in widths)]+[render(row) for row in rows])


def wheel_steps(delta):
    return -int(math.copysign(max(1,abs(delta)//120),delta)) if delta else 0


def open_results_window(entries):
    """One notebook tab; all figures/sections repeat vertically in input order."""
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
    from .viewer import NativeDisplayError

    class Toolbar(NavigationToolbar2Tk):
        toolitems=tuple(item for item in NavigationToolbar2Tk.toolitems if item[0] in ("Home","Back","Forward","Pan","Zoom"))

    root=None;figures=[];closed=False;failures=[]
    try:
        root=tk.Tk();root.title("BANTAI - Detection Results")
        root.geometry(f"{min(1450,int(root.winfo_screenwidth()*.9))}x{min(950,int(root.winfo_screenheight()*.85))}")
        def close():
            nonlocal closed
            closed=True;root.quit()
        def fail(kind,error,traceback):
            failures.append(error);root.quit()
        root.protocol("WM_DELETE_WINDOW",close);root.report_callback_exception=fail
        notebook=ttk.Notebook(root);notebook.pack(fill=tk.BOTH,expand=True)
        tab=ttk.Frame(notebook);notebook.add(tab,text="Detection Results")
        tab.rowconfigure(0,weight=1);tab.columnconfigure(0,weight=1)
        outer=tk.Canvas(tab,highlightthickness=0)
        outer.grid(row=0,column=0,sticky="nsew")
        vertical=ttk.Scrollbar(tab,orient=tk.VERTICAL,command=outer.yview)
        horizontal=ttk.Scrollbar(tab,orient=tk.HORIZONTAL,command=outer.xview)
        vertical.grid(row=0,column=1,sticky="ns");horizontal.grid(row=1,column=0,sticky="ew")
        outer.configure(yscrollcommand=vertical.set,xscrollcommand=horizontal.set)
        content=ttk.Frame(outer,padding=16)
        item=outer.create_window((0,0),window=content,anchor="nw")
        content.bind("<Configure>",lambda event:outer.configure(scrollregion=outer.bbox("all")))
        outer.bind("<Configure>",lambda event:outer.itemconfigure(item,width=max(1140,event.width)))
        def wheel(event):
            steps=wheel_steps(getattr(event,"delta",0))
            if getattr(event,"num",None) in (4,5):steps=-1 if event.num==4 else 1
            if event.state & 1:outer.xview_scroll(steps,"units")
            else:outer.yview_scroll(steps,"units")
            return "break"
        # Local to this root and its children; no global hooks survive closure.
        root.bind("<MouseWheel>",wheel);root.bind("<Button-4>",wheel);root.bind("<Button-5>",wheel)
        def readonly_table(parent,headers,rows):
            frame=ttk.Frame(parent);frame.pack(fill=tk.X,pady=(4,12))
            text=tk.Text(frame,wrap=tk.NONE,height=len(rows)+2,font=("Consolas",10),exportselection=False)
            text.insert("1.0",table_text(headers,rows));text.configure(state=tk.DISABLED)
            text.pack(fill=tk.X)
            bar=ttk.Scrollbar(frame,orient=tk.HORIZONTAL,command=text.xview)
            text.configure(xscrollcommand=bar.set);bar.pack(fill=tk.X)
            def select_all(event):
                text.tag_add(tk.SEL,"1.0","end-1c");return "break"
            text.bind("<Control-a>",select_all)
            text.bind("<MouseWheel>",wheel);text.bind("<Button-4>",wheel);text.bind("<Button-5>",wheel)
            return text
        for index,(image,result) in enumerate(entries,1):
            block=ttk.Frame(content);block.pack(fill=tk.X,pady=(0,28))
            ttk.Label(block,text=f"Image #{index}",font=("Segoe UI",16,"bold")).pack(anchor="w",pady=(0,8))
            figure=Figure(figsize=(11,7.5),dpi=100,layout="constrained");figures.append(figure)
            draw_overlays(figure.subplots(),image,result)
            canvas=FigureCanvasTkAgg(figure,master=block)
            canvas.get_tk_widget().pack(fill=tk.X)
            toolbar=Toolbar(canvas,block,pack_toolbar=False);toolbar.update();toolbar.pack(fill=tk.X)
            canvas.draw()
            data=section_data(result)
            ttk.Label(block,text="Detection Summary",font=("Segoe UI",13,"bold")).pack(anchor="w",pady=(12,0))
            ttk.Label(block,text=f"Processing status: {result['status']}",wraplength=1050).pack(anchor="w")
            if result.get("error"):
                ttk.Label(block,text="Processing error: "+str(result["error"]),wraplength=1050).pack(anchor="w")
            ttk.Label(block,text="Display status is recorded separately after window closure; saved predictions remain intact on display failure.",wraplength=1050).pack(anchor="w")
            if data["summary"]:
                readonly_table(block,("Query","Raw decoded phrase","Mapped category","Mapping reason","Model score","Box xyxy (px)","Box threshold"),data["summary"])
                ttk.Label(block,text="Scores are not accuracy. Full phrases remain here when an image label cannot fit at readable size. Unmapped is not a failed detection.",wraplength=1050).pack(anchor="w")
            else:
                message=("Processing failed; this is not a successful empty detection result."
                         if result["status"]=="error" else
                         "No valid retained detections pass the box threshold. Original image displayed without boxes.")
                ttk.Label(block,text=message).pack(anchor="w")
            ttk.Label(block,text="Text-Token Details",font=("Segoe UI",13,"bold")).pack(anchor="w",pady=(12,0))
            if data["tokens"]:readonly_table(block,("Query","Token","Position","Token ID","Score","Text threshold"),data["tokens"])
            else:ttk.Label(block,text=data["token_message"]).pack(anchor="w")
            ttk.Label(block,text="Configuration",font=("Segoe UI",13,"bold")).pack(anchor="w",pady=(12,0))
            readonly_table(block,("Setting","Actual saved value"),data["configuration"])
        root.mainloop()
        if failures:raise NativeDisplayError(f"Scrollable viewer callback failed ({type(failures[0]).__name__}): {failures[0]}") from failures[0]
        if not closed:raise NativeDisplayError("Scrollable viewer ended without normal window closure")
    finally:
        try:
            for figure in figures:
                figure.canvas.toolbar=None
                figure.clear()
        finally:
            if root is not None:root.destroy()
