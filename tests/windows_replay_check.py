"""Actual Tk protocol checks; not a human visual acceptance test."""
import contextlib,io,json,socket,sys
from pathlib import Path
from unittest.mock import patch
import tkinter as tk
from tkinter import ttk
sys.path.insert(0,str(Path.cwd()))
from scripts.detect_images import main
rp=[Path('results/a2ef6836b41b4e18a793642b1ad461a4/da4b761153fda0359330cde3950d3b4cc1a2247db5bbe6618aec59c092e4afc0.json'),
    Path('results/ea5da8d9bdc840bb924d5b20c2f1733c/9d75f19ceb224a47917ee88b751c4b74928536296000016824365d6cde7787e6.json')]
ips=[Path(json.loads(p.read_text())['source']['image_id']) for p in rp]
before=[p.read_bytes() for p in rp+ips]
real_tk=tk.Tk;observations=[];errors=[]
def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)
def create():
    root=real_tk()
    def probe():
        try:
            widgets=list(descendants(root))
            assert len([w for w in widgets if isinstance(w,ttk.Notebook)])==1
            notebook=next(w for w in widgets if isinstance(w,ttk.Notebook))
            assert len(notebook.tabs())==1
            headings=[w.cget('text') for w in widgets if isinstance(w,ttk.Label) and w.cget('text') in ('Image #1','Image #2','Detection Summary','Text-Token Details','Configuration')]
            assert headings==['Image #1','Detection Summary','Text-Token Details','Configuration','Image #2','Detection Summary','Text-Token Details','Configuration']
            texts=[w for w in widgets if isinstance(w,tk.Text)]
            assert len(texts)==4 and all(str(w.cget('state'))=='disabled' for w in texts)
            assert 'a group' in texts[0].get('1.0','end-1c') and 'discarded item' in texts[0].get('1.0','end-1c')
            root.geometry('850x650');root.update()
            outer=next(w for w in widgets if isinstance(w,tk.Canvas) and w.cget('yscrollcommand'))
            outer.yview_moveto(0);outer.xview_moveto(0);root.update()
            old=outer.yview()[0]
            root.event_generate('<MouseWheel>',delta=-120,state=0);root.update()
            assert outer.yview()[0]>old,('vertical wheel',old,outer.yview())
            old=outer.xview()[0]
            root.event_generate('<MouseWheel>',delta=-120,state=1);root.update()
            assert outer.xview()[0]>old,('horizontal wheel',old,outer.xview())
            outer.yview_moveto(1);outer.xview_moveto(1);root.update_idletasks()
            assert outer.yview()[1]>=.99 and outer.xview()[1]>=.99
            text=texts[0];text.tag_add('sel','1.0','end-1c')
            assert text.tag_ranges('sel')
            try:clipboard=root.clipboard_get()
            except tk.TclError:clipboard=None
            if clipboard is not None:
                try:
                    text.event_generate('<<Copy>>');root.update_idletasks()
                    assert root.clipboard_get()==text.get('1.0','end-1c')
                    observations.append('Native read-only table selection/copy passed; prior text clipboard restored.')
                finally:root.clipboard_clear();root.clipboard_append(clipboard)
            else:observations.append('Selection passed; clipboard copy skipped to preserve an unavailable/non-text clipboard.')
            observations.append('One tab; two complete blocks; native vertical/Shift-wheel/horizontal scrollbars passed at 850x650.')
        except BaseException:
            import traceback
            errors.append(traceback.format_exc())
        finally:root.tk.call(root.protocol('WM_DELETE_WINDOW'))
    root.after(1800,probe)
    return root
args=['--mode','replay']
for result,image in zip(rp,ips):args+=['--result',str(result),'--image',str(image)]
out=io.StringIO()
with patch('tkinter.Tk',side_effect=create),patch.object(socket.socket,'connect',side_effect=AssertionError('No network')) as sockets,contextlib.redirect_stdout(out):
    code=main(args)
assert not errors,repr(errors)
assert code==0 and sockets.call_count==0
assert [p.read_bytes() for p in rp+ips]==before
assert not any(m in sys.modules for m in ('torch','transformers','scripts.inference','scripts.pipeline'))
print('\n'.join(observations))
print(out.getvalue())
print('Actual Tk callbacks/closure passed; no model imports/inference/network; original photo/result bytes unchanged. This is software evidence, not visual proof.')
