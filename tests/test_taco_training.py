"""Controlled generated photographs/metadata only; no real TACO photo access."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor

from scripts.configuration import ROOT
from scripts.training_smoke import LABELS, SNAPSHOT, digest, synthetic_batch, move_batch
from scripts.taco_training_data import (dataset_context, require_rights, RightsRequired,
                                       CocoTrainingDataset, collate, get_batch)
from scripts.training_checkpoint import (save_checkpoint, verify_checkpoint, restore_training,
                                         epoch_batches, rng_state, restore_rng)
from scripts.taco_evaluation import coco_metrics, annotation_errors, decode_predictions
from scripts.train_taco import main


def fixture(folder):
    root=folder/'images';root.mkdir()
    source=folder/'original_annotations.json';source.write_text(json.dumps({'synthetic_fixture':True}))
    manifest={'schema_version':'bantai.taco-preparation.v1','status':'validated',
              'source':{'sha256':digest(source),'annotations':str(source),
                        'taxonomy_sha256':digest(ROOT/'config/taco_taxonomy.v1.json')},'artifacts':{}}
    for split,image_id in (('train',1),('validation',2)):
        path=root/f'{split}.png'
        with Image.new('RGB',(40,30),'orange') as im:im.save(path)
        data={'info':{'image_root':str(root),'coordinate_frame':'stored_raster_without_exif_transpose'},
              'images':[{'id':image_id,'file_name':path.name,'width':40,'height':30,
                         'source_taco':{'file_sha256':digest(path)}}],
              'categories':[{'id':i,'name':name} for i,name in enumerate(LABELS)],
              'annotations':[{'id':image_id,'image_id':image_id,'category_id':0,'bbox':[2,3,8,6],
                              'area':27.,'iscrowd':0}]}
        target=folder/(split+'.json');target.write_text(json.dumps(data))
        manifest['artifacts'][target.name]=digest(target)
    (folder/'preparation_manifest.json').write_text(json.dumps(manifest))
    context=dataset_context(folder)
    rights={'schema_version':'bantai.training-rights.v1','approved':True,'dataset':context,
            'basis_type':'documented_permission','scope':'local_academic_train_validation_no_publication',
            'photo_rights_cleared':True,'annotation_database_rights_cleared':True,
            'approving_authority':'Synthetic fixture creator; NOT real TACO authorization',
            'approval_date':'2026-10-19','basis_and_evidence':'Generated solid-color fixture; not a source photograph',
            'obligations':'Controlled test only'}
    rp=folder/'rights.json';rp.write_text(json.dumps(rights))
    return rp,context


class DataTests(unittest.TestCase):
    def test_source_formatting_equivalence_requires_exact_original_hash_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);_,context=fixture(folder)
            source=folder/'original_annotations.json'
            source.write_text(json.dumps(json.loads(source.read_text()),indent=4))
            observed=dataset_context(folder)
            self.assertEqual(observed['source_annotations_sha256'],context['source_annotations_sha256'])
            self.assertEqual(observed['source_verification'],'exact_recorded_bytes_reconstructed_by_json_dumps')
            self.assertEqual(observed['observed_source_annotations_sha256'],digest(source))
            source.write_text(json.dumps({'synthetic_fixture':False},indent=4))
            with self.assertRaisesRegex(ValueError,'annotation JSON changed'):dataset_context(folder)

    def test_owner_exception_record_preserves_no_clearance_and_binds_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);rp,context=fixture(folder)
            record=json.loads(rp.read_text())
            record.update(schema_version='bantai.training-rights.v2',basis_type='owner_authorized_section185_research',
                workflow_authorized=True,basis_assessment='defensible_limited_use_with_uncertainty',
                rightsholder_permission='not_claimed',institutional_approval='not_claimed',definitive_legal_clearance='not_claimed',
                photo_rights_cleared=False,annotation_database_rights_cleared=False,
                owner_declaration={'non_commercial_academic':True,'local_only':True,
                    'no_dataset_publication_redistribution_upload':True,'workflow_authorized':True},
                assessment={'path':'docs/TACO_STAGE_4_ACADEMIC_USE.md','sha256':digest(ROOT/'docs/TACO_STAGE_4_ACADEMIC_USE.md')},
                factors={k:'Controlled fixture analysis' for k in ('purpose','nature','amount','market')},
                sources=['https://lawphil.net/statutes/repacts/ra1997/ra_8293_1997.html',
                    'https://lawphil.net/statutes/repacts/ra2013/ra_10372_2013.html',
                    'https://github.com/pedropro/TACO/blob/master/README.md'])
            rp.write_text(json.dumps(record));self.assertFalse(require_rights(rp,context)['record']['photo_rights_cleared'])
            for field,value in [('workflow_authorized',False),('photo_rights_cleared',True),('institutional_approval','invented'),
                                ('factors',{}),('assessment',{}),('sources',[])]:
                bad=dict(record,**{field:value});rp.write_text(json.dumps(bad))
                with self.subTest(field=field),self.assertRaises(RightsRequired):require_rights(rp,context)

    def test_missing_dimension_and_unsafe_source_paths_fail(self):
        for change in ('missing','dimensions','escape'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);rp,_=fixture(folder)
                ds=CocoTrainingDataset(folder,'train',rp)
                if change=='missing':(ds.root/'train.png').unlink()
                elif change=='dimensions':ds.images[0]['width']=41
                else:ds.images[0]['file_name']='../outside.png'
                with self.assertRaises(ValueError):ds[0]

    def test_rights_missing_incomplete_or_wrong_snapshot_block_before_image_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);rp,context=fixture(folder)
            with patch('PIL.Image.open',side_effect=AssertionError('must not open')):
                with self.assertRaises(RightsRequired):CocoTrainingDataset(folder,'train',None)
                for field,value in [('approved',False),('photo_rights_cleared',False),
                    ('annotation_database_rights_cleared',False),('dataset',{}),('basis_and_evidence','')]:
                    record=json.loads(rp.read_text());record[field]=value
                    bad=folder/'bad.json';bad.write_text(json.dumps(record))
                    with self.subTest(field=field),self.assertRaises(RightsRequired):
                        require_rights(bad,context)

    def test_test_split_inaccessible_and_no_test_metadata_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);rp,_=fixture(folder)
            (folder/'test.json').write_text('not even valid JSON')
            self.assertEqual(dataset_context(folder)['splits']['train']['images'],1)
            with self.assertRaisesRegex(ValueError,'Test split'):CocoTrainingDataset(folder,'test',rp)

    def test_segmentation_area_preserved_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);rp,_=fixture(folder)
            ds=CocoTrainingDataset(folder,'train',rp)
            image,target=ds[0]
            try:self.assertEqual(target['annotations'][0]['area'],27.)
            finally:image.close()
            (ds.root/'train.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):ds[0]

    def test_malformed_annotations_rejected_without_filtering(self):
        for change in ({'bbox':[-1,0,2,2]},{'bbox':[0,0,0,1]},{'category_id':8},
                       {'iscrowd':1},{'ignore':1},{'image_id':999},{'area':0}):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);rp,_=fixture(folder)
                data=json.loads((folder/'train.json').read_text());data['annotations'][0].update(change)
                (folder/'train.json').write_text(json.dumps(data))
                manifest=json.loads((folder/'preparation_manifest.json').read_text())
                manifest['artifacts']['train.json']=digest(folder/'train.json')
                (folder/'preparation_manifest.json').write_text(json.dumps(manifest))
                record=json.loads(rp.read_text());record['dataset']=dataset_context(folder);rp.write_text(json.dumps(record))
                with self.assertRaises(ValueError):CocoTrainingDataset(folder,'train',rp)

    def test_split_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);rp,_=fixture(folder)
            (folder/'train.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):CocoTrainingDataset(folder,'train',rp)

    def test_image_resources_closed_on_partial_batch_failure(self):
        image=Image.new('RGB',(4,4));closed=[]
        class Dataset:
            def __getitem__(self,index):
                if index:raise ValueError('fixture failure')
                return image,{}
        with patch.object(image,'close',side_effect=lambda:closed.append(True)):
            with self.assertRaises(ValueError):get_batch(Dataset(),[0,1],None,256,384)
        self.assertEqual(closed,[True]);image.close()

    def test_real_cli_rights_gate_never_loads_model_or_photos(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);fixture(folder)
            with (patch('scripts.train_taco.load_model',side_effect=AssertionError('model must not load')),
                  patch('PIL.Image.open',side_effect=AssertionError('photo must not open'))):
                self.assertEqual(main(['--dataset',str(folder)]),2)


@unittest.skipUnless((SNAPSHOT/'preprocessor_config.json').exists(),'Pinned processor unavailable')
class ProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor=AutoProcessor.from_pretrained(SNAPSHOT,local_files_only=True)

    def test_all_classes_fixed_padding_geometry_and_inverse_decode(self):
        images,targets=synthetic_batch(2)
        targets[0]['annotations'][0]['area']=1.  # Valid COCO segmentation area, not bbox area.
        try:
            batch,frames=collate(self.processor,list(zip(images,targets)),256,384)
            self.assertEqual(tuple(batch['pixel_values'].shape),(2,3,384,384))
            self.assertEqual(batch['labels'][0]['class_labels'].tolist(),list(range(8)))
            # Put one confident query exactly on each processed target box.
            from types import SimpleNamespace
            from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps
            logits=torch.full((2,900,256),float('-inf'))
            logits[:,:,:30]=-30
            boxes=torch.full((2,900,4),.001)
            maps=build_label_maps(logits,batch['input_ids'])
            for i in range(2):
                for j,label in enumerate(batch['labels'][i]['class_labels']):
                    logits[i,j,maps[i][label].bool()]=10
                    boxes[i,j]=batch['labels'][i]['boxes'][j]
            preds,raw=decode_predictions(SimpleNamespace(logits=logits,pred_boxes=boxes),batch,frames,self.processor.tokenizer)
            confident=[p for p in preds if p['score']>.99]
            self.assertEqual(len(confident),11)
            for a in targets[0]['annotations']+targets[1]['annotations']:
                matches=[p for p in confident if p['image_id']==a['image_id']
                         and p['category_id']==a['category_id'] and np.allclose(p['bbox'],a['bbox'],atol=4e-5)]
                self.assertEqual(len(matches),1)
            self.assertEqual(len(raw),1800)
        finally:
            for image in images:image.close()

    def test_empty_target_and_extreme_aspect_ratio(self):
        image=Image.new('RGB',(1000,30))
        try:
            batch,_=collate(self.processor,[(image,{'image_id':1,'annotations':[]})],256,384)
            self.assertEqual(batch['labels'][0]['boxes'].shape,(0,4))
            self.assertEqual(tuple(batch['pixel_values'].shape[-2:]),(384,384))
        finally:image.close()


@unittest.skipUnless(importlib.util.find_spec('pycocotools'),'Stage 4 evaluator is training-environment only')
class MetricTests(unittest.TestCase):
    def test_perfect_wrong_and_empty_metrics_and_undefined_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);fixture(folder)
            data=json.loads((folder/'validation.json').read_text())
            good={'image_id':2,'category_id':0,'bbox':[2,3,8,6],'score':.9}
            self.assertAlmostEqual(coco_metrics(data,[good])['AP'],1.)
            self.assertEqual(coco_metrics(data,[])['AP'],0.)
            wrong=dict(good,category_id=1)
            self.assertEqual(coco_metrics(data,[wrong])['AP'],0.)
            self.assertIsNone(coco_metrics(data,[])['classes']['drink_can']['AP'])
            errors=annotation_errors(data,[good,dict(good,score=.8),wrong])[0]['issues']
            self.assertCountEqual([r['kind'] for r in errors],['duplicate','wrong_class'])


class CheckpointTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('pycocotools') and (SNAPSHOT/'preprocessor_config.json').exists(),
                         'Stage 4 evaluator/processor unavailable')
    def test_epoch_loop_and_mid_epoch_resume_on_generated_images_with_toy_updates(self):
        """Real loader/processor/checkpoint orchestration; NOT native-model evidence."""
        import scripts.train_taco as trainer
        from safetensors.torch import save_file,load_file
        processor=AutoProcessor.from_pretrained(SNAPSHOT,local_files_only=True)
        class Model(torch.nn.Linear):
            def save_pretrained(self,folder,**kwargs):
                save_file(self.state_dict(),str(folder/'model.safetensors'))
                (folder/'config.json').write_text('{}')
        class Adapter:
            def manifest(self,model):return {'controlled_orchestration_fixture':True}
            def verify_manifest(self,model,value):assert value==self.manifest(model)
        def load(path,device):
            model=Model(2,1)
            model.load_state_dict(load_file(str(path/'model.safetensors')))
            return model,processor,Adapter()
        def update(model,adapter,optimizer,batch):
            optimizer.zero_grad();loss=model(torch.rand(2,2)).square().mean()
            loss.backward();optimizer.step()
            return {'loss':float(loss.detach())}
        def evaluation(*args):return {'metrics':{'AP':0.},'mean_loss_per_image':1.}
        reports=[]
        original_atomic=trainer.atomic_json
        def capture(path,record,**kwargs):
            if path.name=='report.json' and not kwargs.get('replace'):reports.append(path)
            return original_atomic(path,record,**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);rp,_=fixture(folder)
            base=folder/'base';base.mkdir();Model(2,1).save_pretrained(base)
            argv=['--dataset',str(folder),'--rights',str(rp),'--device','cpu','--pilot-steps','2','--checkpoint-every','1']
            with (patch.object(trainer,'SNAPSHOT',base),patch.object(trainer,'load_model',side_effect=load),
                  patch.object(trainer,'require_training_environment',return_value={
                      'packages':{'fixture':'1'},'python':'fixture','platform':'fixture'}),
                  patch.object(trainer,'train_step',side_effect=update),patch.object(trainer,'evaluate',side_effect=evaluation),
                  patch.object(trainer,'atomic_json',side_effect=capture)):
                self.assertEqual(main(argv),0)
                first=reports[-1].parent
                checkpoint=first/'checkpoints/step-000001'
                self.assertEqual(main(argv+['--resume',str(checkpoint)]),0)
                second=reports[-1].parent
            one=load_file(str(first/'checkpoints/epoch-02/model.safetensors'))
            two=load_file(str(second/'checkpoints/epoch-02/model.safetensors'))
            self.assertTrue(all(torch.equal(one[k],two[k]) for k in one))
            report=json.loads((second/'report.json').read_text())
            self.assertEqual(report['status'],'completed')
            self.assertEqual(report['optimizer_steps'],1)

    def test_rng_and_sampler_resume_exactly(self):
        random.seed(5);np.random.seed(5);torch.manual_seed(5)
        state=rng_state()
        expected=(random.random(),np.random.rand(),torch.rand(3))
        restore_rng(state)
        actual=(random.random(),np.random.rand(),torch.rand(3))
        self.assertEqual(expected[:2],actual[:2]);self.assertTrue(torch.equal(expected[2],actual[2]))
        batches=epoch_batches(7,2,42,0)
        self.assertEqual(sorted(sum(batches,[])),list(range(7)))
        self.assertEqual(batches,epoch_batches(7,2,42,0))
        self.assertNotEqual(batches,epoch_batches(7,2,42,1))

    def test_complete_optimizer_resume_matches_uninterrupted_update_and_rejects_tamper(self):
        class Model(torch.nn.Linear):
            def save_pretrained(self,folder,**kwargs):
                from safetensors.torch import save_file
                save_file(self.state_dict(),str(folder/'model.safetensors'))
                (folder/'config.json').write_text('{}')
        class Processor:
            def save_pretrained(self,folder):(folder/'preprocessor_config.json').write_text('{}')
        class Adapter:
            def manifest(self,model):return {'fixture':True}
            def verify_manifest(self,model,value):assert value==self.manifest(model)
        model=Model(2,1);opt=torch.optim.AdamW(model.parameters(),lr=.01)
        def step(m,o):
            o.zero_grad();loss=m(torch.rand(3,2)).square().mean();loss.backward();o.step()
            return float(loss)
        step(model,opt)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'checkpoint';context={'fixture':True};pos={'epoch':0,'next_batch':1,'global_step':1}
            save_checkpoint(folder,model,Processor(),Adapter(),opt,context,pos)
            expected_loss=step(model,opt);expected=copy.deepcopy(model.state_dict())
            from safetensors.torch import load_file
            restored=Model(2,1);restored.load_state_dict(load_file(str(folder/'model.safetensors')))
            optimizer=torch.optim.AdamW(restored.parameters(),lr=.01)
            self.assertEqual(restore_training(folder,restored,Processor(),Adapter(),optimizer,context),pos)
            self.assertEqual(expected_loss,step(restored,optimizer))
            self.assertTrue(all(torch.equal(expected[k],restored.state_dict()[k]) for k in expected))
            with self.assertRaises(ValueError):verify_checkpoint(folder,{'changed':True})
            (folder/'training_state.pt').write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):verify_checkpoint(folder,context)


if __name__=='__main__':unittest.main()
