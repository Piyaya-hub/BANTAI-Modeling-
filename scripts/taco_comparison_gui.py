"""Standalone in-memory four-view diagnostic TkAgg replay; never model inference."""
import json
import uuid

from .gui_viewer import phrase_artist
from .persistence import atomic_json
from .taco_error_analysis import MODELS,iou
from .training_smoke import LABELS
from .taco_diagnostic_nms import NMS_MODES,diagnostic_nms
from .taco_material_hierarchy import TAXONOMY_VIEWS,map_row,row_label,scheme_key

COLORS=('#ffcc00','#00ffff','#ff66ff','#00ff55','#ff8800','#aaaaFF','#ff5555','#eeeeee')


def visible_rows(rows,threshold,source,nms_mode=NMS_MODES[0],taxonomy='8-specific'):
    if not 0<=threshold<=1:raise ValueError('Diagnostic threshold must be in [0,1]')
    passing=[r for r in rows if r['valid_clipped_box'] and r['score']>=threshold
             and (source=='all 900 (diagnostic)' or r['retained'])]
    if scheme_key(taxonomy)!='8-specific':passing=[map_row(r,taxonomy) for r in passing]
    return diagnostic_nms(passing,nms_mode)


def draw_panel(ax,image,rows,*,ground_truth=False,font_size=5):
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter,MaxNLocator
    width,height=image.size
    ax.imshow(image,origin='upper',extent=(0,width,height,0),interpolation='antialiased')
    labels=[]
    for row in rows:
        x,y,w,h=row['bbox'];category=row['category_id']
        ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=COLORS[category],
            linestyle='--' if ground_truth else '-',linewidth=1.1,zorder=3))
        labels.append({'id':'GT'+str(row['id']) if ground_truth else 'q'+str(row['query_id']),
            'raw_phrase':row_label(row)+('' if ground_truth else f" s={row['score']:.4f}"),
            'box_xyxy':[x,y,x+w,y+h]})
    for label in labels:
        phrase_artist(ax,label,labels,{'label_font_size':font_size,'label_min_font_size':font_size})
    ax.set(xlim=(0,width),ylim=(height,-height*.22),xlabel='x (original pixels)',ylabel='y (original pixels)')
    ax.set_aspect('equal',adjustable='box')
    for axis,maximum in ((ax.xaxis,width),(ax.yaxis,height)):
        axis.set_major_locator(MaxNLocator(nbins=6,integer=True))
        axis.set_major_formatter(FuncFormatter(lambda value,pos,m=maximum:f'{value:g}' if 0<=value<=m else ''))


class ComparisonWindow:
    def __init__(self,dataset,models,sample,folder):
        import tkinter as tk
        from tkinter import ttk,messagebox
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
        self.dataset=dataset;self.models=models;self.sample=sample;self.folder=folder
        self.root=tk.Tk();self.root.title('BANTAI — local validation diagnostic replay')
        self.root.geometry('1400x950');self.closed=False;self.current_image=None;self.error=None
        self.root.report_callback_exception=self.callback_error
        self.index={r['id']:i for i,r in enumerate(dataset.images)}
        self.threshold=tk.StringVar(value='0.05');self.top_k=tk.StringVar(value='5')
        self.source=tk.StringVar(value='official top100');self.font=tk.StringVar(value='5')
        self.nms_mode=tk.StringVar(value=NMS_MODES[0])
        self.taxonomy_view=tk.StringVar(value=TAXONOMY_VIEWS[0])
        self.scope=tk.StringVar(value='representative subset');self.selected=tk.StringVar()
        controls=ttk.Frame(self.root);controls.pack(fill='x')
        ttk.Button(controls,text='Previous',command=lambda:self.navigate(-1)).pack(side='left')
        ttk.Button(controls,text='Next',command=lambda:self.navigate(1)).pack(side='left')
        ttk.Button(controls,text='Reset view',command=lambda:self.render()).pack(side='left')
        scope=ttk.Combobox(controls,textvariable=self.scope,values=['representative subset','all validation'],state='readonly',width=22)
        scope.pack(side='left');scope.bind('<<ComboboxSelected>>',lambda e:self.set_scope())
        self.selector=ttk.Combobox(controls,textvariable=self.selected,state='readonly',width=10)
        self.selector.pack(side='left');self.selector.bind('<<ComboboxSelected>>',lambda e:self.render())
        for label,variable,start,end in [('View threshold',self.threshold,0,1),('Box top K',self.top_k,1,100),('Font pt',self.font,3,12)]:
            ttk.Label(controls,text=label).pack(side='left')
            ttk.Spinbox(controls,textvariable=variable,from_=start,to=end,increment=.01 if variable is self.threshold else 1,width=6).pack(side='left')
        ttk.Combobox(controls,textvariable=self.source,values=['official top100','all 900 (diagnostic)'],state='readonly',width=23).pack(side='left')
        ttk.Button(controls,text='Apply view settings',command=self.apply).pack(side='left')
        nms_controls=ttk.Frame(self.root);nms_controls.pack(fill='x')
        ttk.Label(nms_controls,text='Diagnostic NMS').pack(side='left')
        self.nms_selector=ttk.Combobox(nms_controls,textvariable=self.nms_mode,values=NMS_MODES,state='readonly',width=39)
        self.nms_selector.pack(side='left');self.nms_selector.bind('<<ComboboxSelected>>',lambda e:self.apply())
        ttk.Label(nms_controls,text='Taxonomy view').pack(side='left')
        self.taxonomy_selector=ttk.Combobox(nms_controls,textvariable=self.taxonomy_view,values=TAXONOMY_VIEWS,state='readonly',width=48)
        self.taxonomy_selector.pack(side='left');self.taxonomy_selector.bind('<<ComboboxSelected>>',lambda e:self.apply())
        self.status=tk.StringVar();ttk.Label(self.root,textvariable=self.status,wraplength=1380).pack(fill='x')
        ttk.Label(self.root,text='DIAGNOSTIC settings only; official AP is unchanged. Shared zoom/pan applies to all views. Labels may defer to copyable tables. Unmatched predictions require human review.').pack(fill='x')
        pane=ttk.Panedwindow(self.root,orient='vertical');pane.pack(fill='both',expand=True)
        plot=ttk.Frame(pane);details=ttk.Frame(pane);pane.add(plot,weight=3);pane.add(details,weight=1)
        self.figure=Figure(figsize=(13,7),dpi=100)
        self.axes=list(self.figure.subplots(2,2,sharex=True,sharey=True).flat)
        self.figure.subplots_adjust(left=.07,right=.97,bottom=.08,top=.93,hspace=.32,wspace=.16)
        self.canvas=FigureCanvasTkAgg(self.figure,master=plot);self.canvas.get_tk_widget().pack(fill='both',expand=True)
        class DiagnosticToolbar(NavigationToolbar2Tk):
            toolitems=tuple(item for item in NavigationToolbar2Tk.toolitems if item[0]!='Save')
        self.toolbar=DiagnosticToolbar(self.canvas,plot);self.toolbar.update()
        self.scroll_connection=self.canvas.mpl_connect('scroll_event',self.wheel_zoom)
        notebook=ttk.Notebook(details);notebook.pack(fill='both',expand=True);self.notebook=notebook
        self.tables={}
        columns=('identity','rank','evaluation_class','score','raw_phrase','bbox_xywh','best_GT','IoU','hazard_flag')
        for name in ('ground_truth',)+MODELS:
            frame=ttk.Frame(notebook);notebook.add(frame,text=name)
            tree=ttk.Treeview(frame,columns=columns,show='headings',selectmode='extended',height=6)
            for col in columns:tree.heading(col,text=col);tree.column(col,width=160 if col not in ('raw_phrase','bbox_xywh') else 280,stretch=False)
            y=ttk.Scrollbar(frame,orient='vertical',command=tree.yview);x=ttk.Scrollbar(frame,orient='horizontal',command=tree.xview)
            tree.configure(yscrollcommand=y.set,xscrollcommand=x.set)
            tree.grid(row=0,column=0,sticky='nsew');y.grid(row=0,column=1,sticky='ns');x.grid(row=1,column=0,sticky='ew')
            frame.rowconfigure(0,weight=1);frame.columnconfigure(0,weight=1)
            tree.bind('<Control-c>',lambda event,t=tree:self.copy_rows(t))
            if name in MODELS:
                tree.bind('<<TreeviewSelect>>',lambda event,n=name,t=tree:self.highlight(n,t))
            self.tables[name]=tree
        self.note=tk.StringVar()
        notes=ttk.Frame(self.root);notes.pack(fill='x')
        ttk.Label(notes,text='Optional human review note for current image:').pack(side='left')
        ttk.Entry(notes,textvariable=self.note).pack(side='left',fill='x',expand=True)
        ttk.Button(notes,text='Save explicit note',command=self.save_note).pack(side='left')
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        try:self.set_scope()
        except Exception:
            self.close();raise

    def set_scope(self):
        ids=[r['image_id'] for r in self.sample] if self.scope.get()=='representative subset' else list(self.index)
        self.ids=ids;self.selector['values']=[str(i) for i in ids]
        if self.selected.get() not in [str(i) for i in ids]:self.selected.set(str(ids[0]))
        self.render()

    def set_saved_configuration(self,choice,configuration):
        """Expose actual recorded context for explicit saved-run replay only."""
        import tkinter as tk
        from tkinter import ttk
        self.root.title('BANTAI saved validation review — '+choice)
        self.panel_labels=configuration.get('replay_panels',{})
        for index,name in enumerate(MODELS,1):
            self.notebook.tab(index,text=self.panel_labels.get(name,name))
        frame=ttk.Frame(self.notebook);self.notebook.add(frame,text='Saved configuration')
        text=tk.Text(frame,wrap='none',height=6)
        y=ttk.Scrollbar(frame,orient='vertical',command=text.yview)
        x=ttk.Scrollbar(frame,orient='horizontal',command=text.xview)
        text.configure(yscrollcommand=y.set,xscrollcommand=x.set)
        text.grid(row=0,column=0,sticky='nsew');y.grid(row=0,column=1,sticky='ns');x.grid(row=1,column=0,sticky='ew')
        frame.rowconfigure(0,weight=1);frame.columnconfigure(0,weight=1)
        text.insert('1.0',choice+'\n'+json.dumps(configuration,indent=2,ensure_ascii=False))
        text.configure(state='disabled')
        self.render()

    def navigate(self,offset):
        position=self.ids.index(int(self.selected.get()));self.selected.set(str(self.ids[(position+offset)%len(self.ids)]));self.render()

    def apply(self):
        from tkinter import messagebox
        try:self.render(preserve_view=True)
        except (ValueError,KeyError) as error:messagebox.showerror('Invalid diagnostic settings',str(error))

    def active_taxonomy(self):
        return scheme_key(self.taxonomy_view.get()) if hasattr(self,'taxonomy_view') else '8-specific'

    def render(self,preserve_view=False):
        threshold=float(self.threshold.get());top=int(self.top_k.get());font=float(self.font.get())
        if not 0<=threshold<=1 or not 1<=top<=100 or not 3<=font<=12:raise ValueError('Threshold 0..1, top K 1..100, font 3..12')
        image_id=int(self.selected.get());viewport=(self.axes[0].get_xlim(),self.axes[0].get_ylim())
        if self.current_image is not None:self.current_image.close();self.current_image=None
        image,target=self.dataset[self.index[image_id]];self.current_image=image
        if target['image_id']!=image_id:raise ValueError('Source image identity mismatch')
        self.current_id=image_id
        self.selection_overlay=None
        for ax in self.axes:ax.clear()
        taxonomy=self.active_taxonomy()
        annotations=[map_row(a,taxonomy) for a in target['annotations']]
        draw_panel(self.axes[0],image,annotations,ground_truth=True,font_size=font)
        self.axes[0].set_title(f'Ground truth — image {image_id}, {len(annotations)} annotations (dashed)')
        for table in self.tables.values():table.delete(*table.get_children())
        for ann in annotations:
            self.tables['ground_truth'].insert('', 'end',values=(f"GT{ann['id']}",'',row_label(ann),'','not a prediction',str(ann['bbox']),'','',ann.get('hazard_flag','')))
        for ax,name in zip(self.axes[1:],MODELS):
            rows=visible_rows(self.models[name][image_id],threshold,self.source.get(),self.nms_mode.get(),taxonomy)
            draw_panel(ax,image,rows[:top],font_size=font)
            label=getattr(self,'panel_labels',{}).get(name,name)
            ax.set_title(f'{label}: {len(rows)} at view threshold {threshold:g}; drawing first {min(top,len(rows))}')
            if not rows:
                ax.text(.5,.05,'No candidates at this DIAGNOSTIC threshold',transform=ax.transAxes,ha='center',fontsize=9,color='red')
            for row in rows:
                value,ann=max(((iou(row['bbox'],a['bbox']),a) for a in annotations if taxonomy=='8-specific' or a['category_id']==row['category_id']),key=lambda p:p[0],default=(0,None))
                self.tables[name].insert('', 'end',values=(f"q{row['query_id']}",row['rank'],row_label(row),f"{row['score']:.6f}",
                    json.dumps(row['raw_phrase'],ensure_ascii=False),str(row['bbox']),ann['id'] if ann else '',f'{value:.4f}',row.get('hazard_flag','')))
        if preserve_view:self.axes[0].set_xlim(viewport[0]);self.axes[0].set_ylim(viewport[1])
        reasons=next((s['reasons'] for s in self.sample if s['image_id']==image_id),['full validation access'])
        self.status.set(f'Image {image_id} — stored-raster original {image.width} x {image.height}; source hash verified; Taxonomy: {taxonomy}; NMS: {self.nms_mode.get()}; '+ '; '.join(reasons))
        self.canvas.draw()
        if not preserve_view:
            self.toolbar.update();self.toolbar.push_current()

    def wheel_zoom(self,event):
        if event.inaxes not in self.axes or event.xdata is None or event.ydata is None:return
        factor=.8 if event.button=='up' else 1.25
        ax=event.inaxes;x0,x1=ax.get_xlim();y0,y1=ax.get_ylim()
        ax.set_xlim(event.xdata+(x0-event.xdata)*factor,event.xdata+(x1-event.xdata)*factor)
        ax.set_ylim(event.ydata+(y0-event.ydata)*factor,event.ydata+(y1-event.ydata)*factor)
        self.canvas.draw_idle()

    def copy_rows(self,tree):
        rows=[tree.item(i,'values') for i in (tree.selection() or tree.get_children())]
        text='\t'.join(tree['columns'])+'\n'+'\n'.join('\t'.join(map(str,row)) for row in rows)
        self.root.clipboard_clear();self.root.clipboard_append(text)
        return 'break'

    def highlight(self,name,tree):
        selected=tree.selection()
        if not selected:return
        values=tree.item(selected[0],'values')
        if not values:return
        query=int(str(values[0])[1:])
        row=next(r for r in self.models[name][self.current_id] if r['query_id']==query)
        if self.selection_overlay is not None:self.selection_overlay.remove()
        from matplotlib.patches import Rectangle
        x,y,w,h=row['bbox'];ax=self.axes[1+MODELS.index(name)]
        self.selection_overlay=Rectangle((x,y),w,h,fill=False,edgecolor='white',linewidth=2,linestyle=':',zorder=8)
        ax.add_patch(self.selection_overlay);self.canvas.draw_idle()

    def save_note(self):
        text=self.note.get().strip()
        if not text:return
        atomic_json(self.folder/('human-note-'+uuid.uuid4().hex+'.json'),{'image_id':self.current_id,
            'source':'explicit_local_GUI_entry_not_automated_confirmation','text':text,
            'view_settings':{'threshold':self.threshold.get(),'top_k':self.top_k.get(),'source':self.source.get(),'nms_mode':self.nms_mode.get(),'taxonomy':self.active_taxonomy()}})
        self.note.set('')

    def close(self):
        if self.closed:return
        self.closed=True
        if self.current_image is not None:self.current_image.close();self.current_image=None
        self.canvas.mpl_disconnect(self.scroll_connection)
        self.figure.clear();self.canvas.get_tk_widget().destroy();self.root.destroy()

    def callback_error(self,kind,error,traceback):
        # Tk otherwise logs callback failures and leaves a stale image visible.
        self.error=repr(error)
        self.close()
