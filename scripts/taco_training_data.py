"""Read-only Stage 4 COCO data and explicit photo-use authorization gate."""
import copy
import math
from pathlib import Path

from .configuration import ROOT, strict_load
from .training_smoke import LABELS, PROMPT, digest, validate_taxonomy, validate_encoder_capacity
from .taco_dataset import bbox_issues, valid_id, image_path


class RightsRequired(ValueError):
    """No real-image bytes may be consumed before this gate succeeds."""


def dataset_context(folder):
    """Metadata only; deliberately never load test.json or any photographs."""
    folder = Path(folder).resolve()
    manifest = strict_load(folder/'preparation_manifest.json')
    if manifest['schema_version'] != 'bantai.taco-preparation.v1' or manifest['status'] != 'validated':
        raise ValueError('Require validated Stage 2 export')
    if digest(manifest['source']['annotations']) != manifest['source']['sha256']:
        raise ValueError('Original source annotation JSON changed')
    validate_taxonomy()
    if digest(ROOT/'config/taco_taxonomy.v1.json') != manifest['source']['taxonomy_sha256']:
        raise ValueError('Taxonomy changed')
    context = {'manifest_sha256': digest(folder/'preparation_manifest.json'),
               'source_annotations_sha256': manifest['source']['sha256'], 'splits': {}}
    seen = set()
    for split in ('train', 'validation'):
        path = folder/(split+'.json')
        if digest(path) != manifest['artifacts'][path.name]:
            raise ValueError('Split hash mismatch: '+split)
        data = strict_load(path)
        ids = [i['id'] for i in data['images']]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError('Duplicate or overlapping train/validation image IDs')
        seen.update(ids)
        context['splits'][split] = {'sha256': digest(path), 'image_ids': ids,
                                    'images': len(ids), 'annotations': len(data['annotations'])}
    return context


def require_rights(path, context):
    """Validate an owner-supplied record, not a legal determination by software."""
    if path is None or not Path(path).is_file():
        raise RightsRequired('Documented photo AND annotation/database authorization is required; see config/taco_training_rights.example.json')
    record = strict_load(Path(path))
    if (record.get('schema_version') != 'bantai.training-rights.v1'
            or record.get('approved') is not True or record.get('dataset') != context
            or record.get('basis_type') not in ('documented_permission', 'approved_lawful_use')
            or record.get('scope') != 'local_academic_train_validation_no_publication'
            or record.get('photo_rights_cleared') is not True
            or record.get('annotation_database_rights_cleared') is not True):
        raise RightsRequired('Rights record is incomplete or does not cover this exact train/validation snapshot')
    for field in ('approving_authority', 'approval_date', 'basis_and_evidence', 'obligations'):
        if not isinstance(record.get(field), str) or not record[field].strip() or record[field].startswith('REPLACE'):
            raise RightsRequired('Rights record needs actual '+field)
    return {'record_sha256': digest(path), 'record': record}


class CocoTrainingDataset:
    """Stored-raster coordinates; no EXIF transform, clipping, filtering or exports."""
    def __init__(self, folder, split, rights_path):
        if split not in ('train', 'validation'):
            raise ValueError('Test split is deliberately unavailable to the Stage 4 trainer')
        self.context = dataset_context(folder)
        self.rights = require_rights(rights_path, self.context)
        self.data = strict_load(Path(folder)/(split+'.json'))
        if [(c['id'], c['name']) for c in self.data['categories']] != list(enumerate(LABELS)):
            raise ValueError('COCO category order differs from approved taxonomy')
        self.root = Path(self.data['info']['image_root']).resolve()
        if self.data['info']['coordinate_frame'] != 'stored_raster_without_exif_transpose':
            raise ValueError('Unqualified annotation coordinate frame')
        self.images = self.data['images']
        self.photo_reads = 0
        self.objects = {i['id']: [] for i in self.images}
        image_map = {i['id']: i for i in self.images}
        seen = set()
        for ann in self.data['annotations']:
            image = image_map.get(ann.get('image_id'))
            if image is None or not valid_id(ann.get('id')) or ann['id'] in seen:
                raise ValueError('Invalid COCO image reference or annotation ID')
            seen.add(ann['id'])
            if (type(ann.get('category_id')) is not int or not 0 <= ann['category_id'] < 8
                    or ann.get('iscrowd') != 0 or ann.get('ignore', 0) != 0
                    or bbox_issues(ann.get('bbox'), image['width'], image['height'])
                    or type(ann.get('area')) not in (int, float)
                    or not math.isfinite(ann['area']) or ann['area'] <= 0):
                raise ValueError('Invalid/ignored/excluded annotation; no silent dropping')
            # COCO segmentation area is preserved; it need not equal bbox w*h.
            self.objects[ann['image_id']].append(ann)
        if any(not rows for rows in self.objects.values()):
            raise ValueError('Unexpected empty image in strict-clean v1')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        from PIL import Image
        row = self.images[index]
        path = image_path(self.root,row['file_name'])
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError('Unsafe or missing source photograph')
        self.photo_reads += 1
        if digest(path) != row['source_taco']['file_sha256']:
            raise ValueError('Source photograph hash mismatch')
        with Image.open(path) as image:
            image.load()
            if image.size != (row['width'], row['height']) or image.getexif().get(274, 1) != 1:
                raise ValueError('Actual photograph dimensions/orientation differ from strict-clean coordinates')
            rgb = image.convert('RGB')
        return rgb, {'image_id': row['id'], 'annotations': copy.deepcopy(self.objects[row['id']])}


def collate(processor, samples, shortest=256, longest=384):
    """Explicit fixed square padding keeps 900 proposals valid for narrow photos."""
    import torch
    from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps
    images, targets = zip(*samples)
    batch = processor(images=list(images), text=[PROMPT]*len(images), annotations=copy.deepcopy(list(targets)),
                      padding=True, return_tensors='pt', size={'shortest_edge': shortest, 'longest_edge': longest},
                      pad_size={'height': longest, 'width': longest})
    ph, pw = batch['pixel_values'].shape[-2:]
    validate_encoder_capacity(ph, pw)
    maps = build_label_maps(torch.zeros(len(images), 1, 256), batch['input_ids'])
    geometry = []
    for i, (image, target, label, mapping) in enumerate(zip(images, targets, batch['labels'], maps)):
        spans = [processor.tokenizer.decode(batch['input_ids'][i][m[:len(batch['input_ids'][i])].bool()]) for m in mapping]
        if spans != [name.replace('_', ' ') for name in LABELS]:
            raise ValueError('Prompt/category span mismatch')
        mask = batch['pixel_mask'][i]
        rw, rh = int(mask[0].sum()), int(mask[:, 0].sum())
        expected = [[(x+w/2)*rw/image.width/pw, (y+h/2)*rh/image.height/ph,
                     w*rw/image.width/pw, h*rh/image.height/ph]
                    for x,y,w,h in (a['bbox'] for a in target['annotations'])]
        if (label['class_labels'].tolist() != [a['category_id'] for a in target['annotations']]
                or not torch.allclose(label['boxes'], torch.tensor(expected).reshape(-1,4), atol=2e-6, rtol=1e-5)):
            raise ValueError('Processor changed target identity/count/coordinates')
        geometry.append({'image_id': target['image_id'], 'original_wh': list(image.size),
                         'resized_wh': [rw,rh], 'padded_wh': [pw,ph]})
    return batch, geometry


def get_batch(dataset, indexes, processor, shortest, longest):
    samples = []
    try:
        for i in indexes:
            samples.append(dataset[i])
        return collate(processor, samples, shortest, longest)
    finally:
        for image, _ in samples:
            image.close()
