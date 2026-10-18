"""Optional actual Windows Tk check on an existing external-data audit report.

Not collected by unittest; no model, network, dataset mutation or image export.
"""
import argparse
import contextlib
import io
from pathlib import Path
import socket
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from tkinter import ttk
from matplotlib.backend_bases import MouseEvent
from scripts import view_taco
from scripts.taco_dataset import bbox_issues, load_report, sha256


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='reports/taco/review-v1')
    parser.add_argument('--frame', choices=('stored', 'exif'), default='stored')
    parser.add_argument('--image-id', type=int)
    args = parser.parse_args()
    audit = load_report(args.report)
    real_tk = tk.Tk
    real_draw = view_taco.draw_annotations
    real_open = view_taco.open_verified_image
    failures = []; figures = []; seen = {}; current = {}

    def open_image(row, frame='stored'):
        seen[row['path']] = row['sha256']
        return real_open(row, frame)

    def draw(ax, image, rows, selected):
        current.update(ax=ax, rows=rows)
        figures.append(ax.figure)
        real_draw(ax, image, rows, selected)
        boxes = [r['original']['bbox'] for r in rows if not bbox_issues(r['original'].get('bbox'), image.width, image.height)]
        assert len(ax.patches) == len(boxes)
        assert [(p.get_x(), p.get_y(), p.get_width(), p.get_height()) for p in ax.patches] == [tuple(b) for b in boxes]

    def create():
        root = real_tk()
        def probe():
            try:
                widgets = list(descendants(root))
                text = next(w for w in widgets if isinstance(w, tk.Text))
                buttons = {w.cget('text'): w for w in widgets if isinstance(w, ttk.Button)}
                combo = next(w for w in widgets if isinstance(w, ttk.Combobox))
                assert str(text.cget('state')) == 'disabled'
                choices = view_taco.select_images(audit, sample_per_class=1,
                                                  image_ids=[args.image_id] if args.image_id is not None else None, frame=args.frame)
                for _ in choices:
                    value = text.get('1.0', 'end-1c')
                    for row in current['rows']:
                        assert str(row['original']['id']) in value
                        assert row['handling'] in value
                    buttons['Next'].invoke(); root.update()
                for selection in ('excluded', 'unlabeled'):
                    combo.set(selection); combo.event_generate('<<ComboboxSelected>>'); root.update()
                    assert selection+'_requires_decision' in text.get('1.0', 'end-1c')
                text.tag_add('sel', '1.0', 'end-1c'); assert text.tag_ranges('sel')
                ax = current['ax']; canvas = ax.figure.canvas; toolbar = canvas.toolbar
                before = (ax.get_xlim(), ax.get_ylim())
                x, y, w, h = ax.bbox.x0, ax.bbox.y0, ax.bbox.width, ax.bbox.height
                def event(name, px, py):
                    canvas.callbacks.process(name, MouseEvent(name, canvas, px, py, button=1))
                toolbar.zoom()
                event('button_press_event', x+w*.2, y+h*.2)
                event('motion_notify_event', x+w*.8, y+h*.8)
                event('button_release_event', x+w*.8, y+h*.8); canvas.draw()
                assert (ax.get_xlim(), ax.get_ylim()) != before
                toolbar.zoom(); toolbar.home(); canvas.draw()
                assert (ax.get_xlim(), ax.get_ylim()) == before
                assert 'Save' not in [item[0] for item in toolbar.toolitems]
            except BaseException:
                import traceback
                failures.append(traceback.format_exc())
            finally:
                root.tk.call(root.protocol('WM_DELETE_WINDOW'))
        root.after(1500, probe)
        return root

    out = io.StringIO()
    source = audit['summary']['source']; before = sha256(source['annotations'])
    with patch('tkinter.Tk', side_effect=create), patch.object(view_taco, 'draw_annotations', side_effect=draw), \
            patch.object(view_taco, 'open_verified_image', side_effect=open_image), \
            patch.object(socket.socket, 'connect', side_effect=AssertionError('network')), contextlib.redirect_stdout(out):
        command = ['--report', args.report, '--sample-per-class', '1', '--frame', args.frame]
        if args.image_id is not None: command += ['--image-id', str(args.image_id)]
        code = view_taco.main(command)
    assert not failures, failures
    assert code == 0, out.getvalue()
    assert all(not f.axes for f in figures)
    assert all(sha256(p) == digest for p, digest in seen.items())
    assert sha256(source['annotations']) == before
    assert not any(m in sys.modules for m in ('torch', 'transformers', 'scripts.inference', 'scripts.pipeline'))
    print(f'Actual Tk passed: {len(seen)} source images, Next/class selection, excluded/unlabeled boxes, xywh geometry, read-only selection, zoom/Home and closure.')
    print('Original hashes unchanged; no model imports/network/image exports. Human visual acceptance remains separate.')


if __name__ == '__main__':
    main()
