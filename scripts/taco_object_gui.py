"""Opt-in object inspection reusing the existing four-panel TkAgg window."""
import json
from .taco_comparison_gui import ComparisonWindow, draw_panel, visible_rows
from .taco_diagnostic_nms import NMS_MODES
from .taco_error_analysis import MODELS
from .taco_object_review import annotation_review, eligible_rows, object_limits
from .training_smoke import LABELS
from .taco_error_analysis import iou
from .taco_material_hierarchy import map_row,row_label


def object_view_review(ann,raw_rows,kept_rows,threshold,source,nms_mode,taxonomy='8-specific'):
    """Post-NMS matching with unchanged raw-query provenance/no-floor diagnostic."""
    if taxonomy!='8-specific':
        ann=map_row(ann,taxonomy);raw_rows=[map_row(r,taxonomy) for r in raw_rows]
        kept_rows=[map_row(r,taxonomy) for r in kept_rows]
    result=annotation_review(ann,raw_rows if nms_mode==NMS_MODES[0] else kept_rows,threshold,source)
    if nms_mode!=NMS_MODES[0]:
        result['stored_queries']=len(raw_rows)
        result['same_class_coverage_all900_no_score_floor']=any(
            r['valid_clipped_box'] and r['category_id']==ann['category_id'] and iou(r['bbox'],ann['bbox'])>=.5
            for r in raw_rows)
        result['diagnostic_nms_mode']=nms_mode
    return result


class PerObjectWindow(ComparisonWindow):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        import tkinter as tk
        from tkinter import ttk
        self.object_id=tk.StringVar();self.object_zoom=tk.BooleanVar(value=True)
        controls=ttk.Frame(self.root);controls.pack(fill='x',before=self.root.winfo_children()[1])
        ttk.Label(controls,text='Per-object diagnostic — GT annotation').pack(side='left')
        self.object_selector=ttk.Combobox(controls,textvariable=self.object_id,state='readonly',width=12)
        self.object_selector.pack(side='left');self.object_selector.bind('<<ComboboxSelected>>',lambda e:self.render())
        ttk.Button(controls,text='Previous GT',command=lambda:self.navigate_object(-1)).pack(side='left')
        ttk.Button(controls,text='Next GT',command=lambda:self.navigate_object(1)).pack(side='left')
        ttk.Checkbutton(controls,text='Zoom to selected GT',variable=self.object_zoom,command=self.render).pack(side='left')
        ttk.Label(controls,text='GT yellow dashed; class agrees green; differs magenta. Tables include all eligible overlaps.').pack(side='left')
        frame=ttk.Frame(self.notebook);self.notebook.add(frame,text='Per-annotation results')
        columns=('GT','GT_class','GT_bbox','model','best_any','any_class','any_score','any_IoU','best_same','same_score','same_IoU','same_ge050','any_ge050','GT_hazard_flag','any_hazard_flag','same_hazard_flag')
        self.object_table=ttk.Treeview(frame,columns=columns,show='headings',selectmode='browse',height=8)
        for col in columns:self.object_table.heading(col,text=col);self.object_table.column(col,width=130,stretch=False)
        y=ttk.Scrollbar(frame,orient='vertical',command=self.object_table.yview);x=ttk.Scrollbar(frame,orient='horizontal',command=self.object_table.xview)
        self.object_table.configure(yscrollcommand=y.set,xscrollcommand=x.set)
        self.object_table.grid(row=0,column=0,sticky='nsew');y.grid(row=0,column=1,sticky='ns');x.grid(row=1,column=0,sticky='ew')
        frame.rowconfigure(0,weight=1);frame.columnconfigure(0,weight=1)
        self.object_table.bind('<Control-c>',lambda e:self.copy_rows(self.object_table))
        self.object_table.bind('<<TreeviewSelect>>',self.select_object_row)
        # Explicit mode defaults to every stored valid query at the same .05 floor.
        self.source.set('all 900 (diagnostic)')
        self.object_mode=True;self.render();self.notebook.select(frame)

    def navigate_object(self,offset):
        ids=list(self.object_selector['values'])
        self.object_id.set(ids[(ids.index(self.object_id.get())+offset)%len(ids)])
        self.render()

    def select_object_row(self,event=None):
        selection=self.object_table.selection()
        if selection:
            value=str(self.object_table.item(selection[0],'values')[0])
            if value!=self.object_id.get():self.object_id.set(value);self.render()

    def save_note(self):
        import uuid
        from .persistence import atomic_json
        text=self.note.get().strip()
        if not text:return
        atomic_json(self.folder/('human-note-'+uuid.uuid4().hex+'.json'),{
            'image_id':self.current_id,'annotation_id':int(self.object_id.get()),
            'source':'explicit_local_GUI_entry_not_automated_confirmation','text':text,
            'run_choice':getattr(self,'saved_run_choice',None),
            'view_settings':{'threshold':self.threshold.get(),'top_k':self.top_k.get(),'source':self.source.get(),'nms_mode':self.nms_mode.get(),'taxonomy':self.active_taxonomy()}})
        self.note.set('')

    def set_saved_configuration(self,choice,configuration):
        self.saved_run_choice=choice
        return super().set_saved_configuration(choice,configuration)

    def render(self,preserve_view=False):
        if not getattr(self,'object_mode',False):return super().render(preserve_view)
        from matplotlib.patches import Rectangle
        threshold=float(self.threshold.get());font=float(self.font.get());top=int(self.top_k.get())
        if not 0<=threshold<=1 or not 3<=font<=12 or not 1<=top<=100:raise ValueError('Invalid diagnostic settings')
        image_id=int(self.selected.get());old_id=getattr(self,'current_id',None)
        viewport=(self.axes[0].get_xlim(),self.axes[0].get_ylim())
        if self.current_image is not None:self.current_image.close()
        image,target=self.dataset[self.index[image_id]];self.current_image=image;self.current_id=image_id
        if target['image_id']!=image_id:raise ValueError('Source image identity mismatch')
        taxonomy=self.active_taxonomy()
        anns=[map_row(a,taxonomy) for a in target['annotations']];ids=[str(a['id']) for a in anns]
        self.object_selector['values']=ids
        if old_id!=image_id or self.object_id.get() not in ids:self.object_id.set(ids[0])
        ann=next(a for a in anns if str(a['id'])==self.object_id.get())
        for ax in self.axes:ax.clear()
        for table in self.tables.values():table.delete(*table.get_children())
        self.object_table.delete(*self.object_table.get_children());self.selection_overlay=None
        for a in anns:self.tables['ground_truth'].insert('','end',values=(f"GT{a['id']}",'',row_label(a),'','not a prediction',str(a['bbox']),'','',a.get('hazard_flag','')))
        draw_panel(self.axes[0],image,[],font_size=font)
        self.axes[0].set_title(f"Image {image_id} — GT{ann['id']} {row_label(ann)}")
        self.object_results={}
        for name,ax in zip(MODELS,self.axes[1:]):
            raw_group=self.models[name][image_id]
            group=visible_rows(raw_group,threshold,self.source.get(),self.nms_mode.get(),taxonomy)
            result=object_view_review(ann,raw_group,group,threshold,self.source.get(),self.nms_mode.get(),taxonomy);self.object_results[name]=result
            overlaps=result['overlapping_candidates']
            # Always prioritize both the best same-class and best differing-class
            # overlaps. Full tables retain every overlap independent of box cap.
            chosen=[]
            for same in (True,False):
                first=next((r for r in overlaps if r['same_class_as_annotation']==same),None)
                if first:chosen.append(first)
            cap=max(2,top)
            for row in overlaps:
                if len(chosen)>=cap:break
                if row not in chosen:chosen.append(row)
            draw_panel(ax,image,[],font_size=font)
            for row in chosen:
                x,y,w,h=row['bbox'];color='#00a050' if row['same_class_as_annotation'] else '#cc00cc'
                ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor=color,linewidth=1,alpha=.8,zorder=4))
            label=getattr(self,'panel_labels',{}).get(name,name)
            ax.set_title(f'{label}: {len(overlaps)} overlaps; {len(chosen)} boxes; same/any ≥.50: {result["correct_class_coverage"]}/{result["any_class_coverage"]}',fontsize=9)
            for row in overlaps:
                relation='agrees with GT' if row['same_class_as_annotation'] else 'differs from GT'
                if taxonomy!='8-specific':relation+=' material'
                self.tables[name].insert('','end',values=(f"q{row['query_id']}",row['rank'],row['predicted_class']+' ('+relation+')',f"{row['score']:.6f}",json.dumps(row['raw_phrase'],ensure_ascii=False),str(row['bbox']),ann['id'],f"{row['iou']:.6f}",row.get('hazard_flag','')))
            for a in anns:
                value=object_view_review(a,raw_group,group,threshold,self.source.get(),self.nms_mode.get(),taxonomy);any_row=value['best_any'];same_row=value['best_same']
                def field(row,key):return '' if row is None else f'{row[key]:.6f}' if key in ('score','iou') else row[key]
                self.object_table.insert('','end',values=(a['id'],row_label(a),str(a['bbox']),label,
                    field(any_row,'query_id'),field(any_row,'predicted_class'),field(any_row,'score'),field(any_row,'iou'),
                    field(same_row,'query_id'),field(same_row,'score'),field(same_row,'iou'),value['correct_class_coverage'],value['any_class_coverage'],
                    a.get('hazard_flag',''),any_row.get('hazard_flag','') if any_row else '',same_row.get('hazard_flag','') if same_row else ''))
        for ax in self.axes:
            x,y,w,h=ann['bbox'];ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#ffcc00',linestyle='--',linewidth=2,zorder=7))
        if preserve_view and image_id==old_id:
            self.axes[0].set_xlim(viewport[0]);self.axes[0].set_ylim(viewport[1])
        elif self.object_zoom.get():
            xlim,ylim=object_limits(ann,image.width,image.height);self.axes[0].set_xlim(xlim);self.axes[0].set_ylim(ylim)
        self.status.set(f'Image {image_id}; selected GT{ann["id"]}; verified original pixels; {self.source.get()}, score ≥{threshold:g}; Taxonomy: {taxonomy}; NMS: {self.nms_mode.get()}. Remapping precedes NMS and GT matching. Green agrees with GT '+('class' if taxonomy=='8-specific' else 'material')+'. Drawing is GT-prioritized. Coverage is non-exclusive; raw queries remain saved.')
        self.canvas.draw();self.toolbar.update();self.toolbar.push_current()
