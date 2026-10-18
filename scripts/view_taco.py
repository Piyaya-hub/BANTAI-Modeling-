"""Local review of existing TACO boxes; sampling never hides excluded annotations."""
import argparse
from collections import defaultdict
import hashlib
import io
from pathlib import Path
import random
import sys
import textwrap

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_taco import add_source_arguments
from scripts.taco_dataset import DEFAULT_DATASET, DEFAULT_TAXONOMY, audit_dataset, bbox_issues, load_report


def aligned(row, frame='stored'):
    if frame not in ('stored', 'exif'):
        raise ValueError('frame must be stored or exif')
    if row['status'] != 'readable' or any(i != 'declared_actual_dimension_mismatch' for i in row['issues']):
        return False
    size = ((row['actual_width'], row['actual_height']) if frame == 'stored'
            else (row['exif_width'], row['exif_height']))
    return size == (row['original']['width'], row['original']['height'])


def select_images(audit, classes=None, sample_per_class=2, seed=42, image_ids=None, frame='stored'):
    """Deterministic per-class sample of usable annotations, not a dataset split."""
    if type(sample_per_class) is not int or sample_per_class < 1:
        raise ValueError('sample_per_class must be positive')
    ready = {r['original']['id']: r for r in audit['images']
             if aligned(r, frame)}
    if image_ids is not None:
        if any(ident not in ready for ident in image_ids):
            raise ValueError('Requested image is missing, unreadable, ambiguous or dimensionally unaligned')
        return sorted(set(image_ids))
    labels = {r['name'] for r in audit['taxonomy']['labels']}
    classes = list(classes) if classes is not None else sorted(labels)
    if not classes or not set(classes) <= labels | {'excluded', 'unlabeled'}:
        raise ValueError('Unknown/empty class selection; use BANTAI labels, excluded or unlabeled')
    rng = random.Random(seed)
    chosen = set()
    for label in sorted(set(classes)):
        eligible = set()
        for row in audit['annotations']:
            if not row['valid_annotation'] or row['original']['image_id'] not in ready:
                continue
            if (row['bantai_label'] == label or label == 'excluded' and row['bantai_label'] is None
                    or label == 'unlabeled' and row['handling'] == 'unlabeled_requires_decision'):
                eligible.add(row['original']['image_id'])
        chosen.update(rng.sample(sorted(eligible), min(sample_per_class, len(eligible))))
    return sorted(chosen)


def open_verified_image(row, frame='stored'):
    from PIL import Image, ImageOps
    if not aligned(row, frame):
        raise ValueError('Declared dimensions do not match the requested coordinate frame')
    payload = Path(row['path']).read_bytes()
    if hashlib.sha256(payload).hexdigest() != row['sha256']:
        raise ValueError('Image bytes changed since audit; generate a new report')
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.size != (row['actual_width'], row['actual_height']):
            raise ValueError('Image dimensions changed since audit')
        if frame == 'exif':
            with ImageOps.exif_transpose(image) as oriented:
                return oriented.convert('RGB')
        return image.convert('RGB')


def annotation_style(row, selected):
    if not row['valid_annotation']:
        return '#c00000', '-.', 'INVALID - review required'
    if row['handling'] == 'unlabeled_requires_decision':
        return '#a000c0', ':', 'UNLABELED - decision required'
    if row['bantai_label'] is None:
        return '#c56800', '--', 'EXCLUDED - decision required'
    if row['bantai_label'] in selected:
        return '#008040', '-', row['bantai_label']
    return '#006dcc', '-', row['bantai_label']


def draw_annotations(ax, image, rows, selected):
    """COCO xywh means rectangle origin x,y and width,height (never xyxy)."""
    from matplotlib.patches import Rectangle
    ax.imshow(image, origin='upper', extent=(0, image.width, image.height, 0))
    for row in rows:
        box = row['original'].get('bbox')
        if bbox_issues(box, image.width, image.height):
            continue  # Retained in the table; unsafe geometry is never silently clipped.
        x, y, width, height = box
        color, linestyle, target = annotation_style(row, selected)
        ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=color, linestyle=linestyle, linewidth=1.5))
        original = row['original_category']
        name = original['name'] if original else f"unknown category {row['original'].get('category_id')}"
        label = f"a{row['original'].get('id', '?')}: {name} -> {target}"
        ax.annotate('\n'.join(textwrap.wrap(label, 48, break_long_words=True)), (x, y), xytext=(1, 3),
                    textcoords='offset points', va='bottom', fontsize=7,
                    color='white', bbox=dict(facecolor=color, edgecolor='none', alpha=.85),
                    annotation_clip=True, clip_on=False, parse_math=False)
    ax.set(xlim=(0, image.width), ylim=(image.height, -image.height*.12),
           xlabel='x (stored raster pixels)', ylabel='y (stored raster pixels)')
    ax.set_aspect('equal', adjustable='box')


def details_text(image_row, rows, frame='stored'):
    lines = [f"Image ID {image_row['original']['id']}: {image_row['original']['file_name']}",
             f"Stored raster {image_row['actual_width']} x {image_row['actual_height']}; EXIF orientation {image_row['exif_orientation']}; preview frame={frame}",
             'EXIF preview is an explicit inspection option; annotation coordinates are not transformed. Coordinate policy requires review.',
             'Existing annotations only; no predictions/scores. All annotations shown, including excluded/unlabeled.',
             'Green: selected class; blue: other mapped; orange dashed: excluded; purple dotted: Unlabeled litter; red: invalid.',
             'Crowded labels may overlap; zoom/pan and use this selectable table. Invalid boxes are listed, not drawn.', '']
    for row in rows:
        original = row['original']; category = row['original_category']
        name = category['name'] if category else 'unknown'
        lines.append(f"a{original.get('id', '?')} | TACO {original.get('category_id')} {name} -> {row['bantai_label'] or '(no mapping)'} | "
                     f"{row['handling']} | COCO xywh={original.get('bbox')} | iscrowd={original.get('iscrowd', 'not recorded')} | "
                     f"area={original.get('area', 'not recorded')} | issues={row['issues']}")
    if not rows:
        lines.append('No annotations recorded for this image; this does not establish a negative training example.')
    return '\n'.join(lines)


def open_window(audit, classes=None, sample_per_class=2, seed=42, image_ids=None, frame='stored'):
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    class Toolbar(NavigationToolbar2Tk):
        toolitems = tuple(i for i in NavigationToolbar2Tk.toolitems if i[0] in ('Home', 'Back', 'Forward', 'Pan', 'Zoom'))

    images = {r['original']['id']: r for r in audit['images'] if aligned(r, frame)}
    by_image = defaultdict(list)
    for row in audit['annotations']:
        original = row['original']
        if isinstance(original, dict) and type(original.get('image_id')) is int:
            by_image[original['image_id']].append(row)
    selected = set(classes or [r['name'] for r in audit['taxonomy']['labels']])
    choices = select_images(audit, selected, sample_per_class, seed, image_ids, frame)
    if not choices:
        raise ValueError('No usable downloaded images for the selected classes')
    root = None; figure = None; failures = []; closed = False; position = 0
    try:
        root = tk.Tk(); root.title('BANTAI - TACO Annotation Review'); root.geometry('1200x900')
        root.rowconfigure(1, weight=1); root.columnconfigure(0, weight=1)
        controls = ttk.Frame(root); controls.grid(row=0, column=0, sticky='ew')
        status = tk.StringVar()
        figure = Figure(figsize=(11, 6), dpi=100, layout='constrained')
        canvas = FigureCanvasTkAgg(figure, master=root); canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew')
        toolbar = Toolbar(canvas, root, pack_toolbar=False); toolbar.grid(row=2, column=0, sticky='ew')
        table_frame = ttk.Frame(root); table_frame.grid(row=3, column=0, sticky='ew')
        table_frame.columnconfigure(0, weight=1)
        table = tk.Text(table_frame, height=10, wrap='none', font=('Consolas', 9), exportselection=False)
        table.grid(row=0, column=0, sticky='ew')
        vertical = ttk.Scrollbar(table_frame, orient='vertical', command=table.yview)
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal = ttk.Scrollbar(table_frame, orient='horizontal', command=table.xview)
        horizontal.grid(row=1, column=0, sticky='ew')
        table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        def select_all(event):
            table.tag_add('sel', '1.0', 'end-1c'); return 'break'
        table.bind('<Control-a>', select_all)
        def render():
            row = images[choices[position]]
            with open_verified_image(row, frame) as image:
                figure.clear(); ax = figure.subplots()
                draw_annotations(ax, image, by_image[choices[position]], selected)
                ax.set_xlabel(f'x ({frame} preview pixels)'); ax.set_ylabel(f'y ({frame} preview pixels)')
                toolbar.update(); canvas.draw()
            status.set(f'{position+1}/{len(choices)} - image {choices[position]}')
            table.configure(state='normal'); table.delete('1.0', 'end')
            table.insert('1.0', details_text(row, by_image[choices[position]], frame)); table.configure(state='disabled')
        def move(step):
            nonlocal position
            if choices:
                position = (position+step) % len(choices); render()
        ttk.Button(controls, text='Previous', command=lambda: move(-1)).pack(side='left')
        ttk.Button(controls, text='Next', command=lambda: move(1)).pack(side='left')
        ttk.Label(controls, textvariable=status).pack(side='left', padx=12)
        selection = tk.StringVar(value='CLI selection')
        combo = ttk.Combobox(controls, textvariable=selection, state='readonly', width=24,
                             values=['CLI selection']+[r['name'] for r in audit['taxonomy']['labels']]+['excluded', 'unlabeled'])
        combo.pack(side='left')
        def resample(event=None):
            nonlocal choices, position, selected
            selected = set(classes or [r['name'] for r in audit['taxonomy']['labels']]) if selection.get() == 'CLI selection' else {selection.get()}
            choices = select_images(audit, selected, sample_per_class, seed,
                                    image_ids if selection.get() == 'CLI selection' else None, frame)
            position = 0
            if choices: render()
            else:
                figure.clear(); toolbar.update(); canvas.draw(); status.set('No usable images for this selection')
                table.configure(state='normal'); table.delete('1.0', 'end'); table.configure(state='disabled')
        combo.bind('<<ComboboxSelected>>', resample)
        def close():
            nonlocal closed
            closed = True; root.quit()
        def fail(kind, error, traceback):
            failures.append(error); root.quit()
        root.report_callback_exception = fail; root.protocol('WM_DELETE_WINDOW', close)
        render(); root.mainloop()
        if failures: raise RuntimeError(f'TACO viewer callback failed: {failures[0]}') from failures[0]
        if not closed: raise RuntimeError('TACO viewer ended without normal closure')
    finally:
        if figure is not None:
            figure.canvas.toolbar = None; figure.clear()
        if root is not None: root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument('--report', help='Existing report directory; verifies annotation/taxonomy identity')
    parser.add_argument('--frame', choices=('stored', 'exif'), default='stored', help='Explicit preview orientation; never changes source annotations')
    parser.add_argument('--classes', nargs='+', help='BANTAI labels (space separated), excluded, or unlabeled')
    parser.add_argument('--sample-per-class', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image-id', type=int, action='append', help='Explicit COCO image ID; repeat to inspect several')
    parser.add_argument('--list', action='store_true', help='List selected images without opening Tk')
    args = parser.parse_args(argv)
    try:
        if args.report and (args.annotations or args.dataset_root != str(DEFAULT_DATASET) or args.taxonomy != str(DEFAULT_TAXONOMY)):
            raise ValueError('--report uses its recorded sources; do not combine it with source overrides')
        audit = load_report(args.report) if args.report else audit_dataset(args.dataset_root, args.annotations, args.taxonomy)
        choices = select_images(audit, args.classes, args.sample_per_class, args.seed, args.image_id, args.frame)
        if not choices: raise ValueError('No usable images for this selection')
        if args.list:
            indexed = {r['original']['id']: r for r in audit['images'] if isinstance(r['original'], dict) and type(r['original'].get('id')) is int}
            for ident in choices: print(ident, indexed[ident]['original']['file_name'])
        else:
            open_window(audit, args.classes, args.sample_per_class, args.seed, args.image_id, args.frame)
            print('TACO annotation viewer closed normally; no images exported.')
        return 0
    except Exception as exc:
        print(f'TACO viewer failed ({type(exc).__name__}): {exc}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
