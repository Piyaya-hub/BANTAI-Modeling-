"""Saved replay guards; fixtures use no real models or photos."""
import contextlib
import io
import json
import tempfile
import unittest
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts import taco_saved_review as replay
from scripts.taco_visual_review import main


class SavedReviewTests(unittest.TestCase):
    def test_completed_run_binding_and_tamper_guards(self):
        class Dataset:
            context={'verified':True}
            rights={'record_sha256':'rights'}
            images=[{'id':42}]
            data={'annotations':list(range(75))}
            def __len__(self):return 47
        ev={'metrics':{'AP':.1},'predictions':[],'raw_queries':[]}
        report={'protocol':replay.PROTOCOL,'contract':{'dataset':Dataset.context,'seed':123,
                 'optimizer':{'lr':1e-5},'rights_sha256':'rights'},
                 'epochs':[{'validation':ev['metrics']}]*2}
        entry={'report':{'path':'report.json'},'seed':123,'learning_rate':1e-5,
               'checkpoints':{f'epoch-{e:02d}':{'path':f'epoch{e}/checkpoint_manifest.json'} for e in (1,2)},
               'evaluations':{f'epoch-{e:02d}':{'path':f'eval{e}.json'} for e in (1,2)}}
        original={'schema_version':'bantai.cross-seed-review.v1','dataset':Dataset.context,
                  'runs':{'x':entry},'baseline':{'path':'shared-baseline.json'},'images':[{'image_id':42,'reasons':[]}]}
        for mutation in (None,'seed','position','metrics','image','baseline'):
            manifest=copy.deepcopy(original);saved_report=copy.deepcopy(report)
            if mutation=='seed':manifest['runs']['x']['seed']=42
            if mutation=='image':manifest['images'][0]['image_id']=99
            def read(path):
                name=str(path)
                if name=='manifest':return manifest
                if name=='report.json':return saved_report
                value=copy.deepcopy(ev)
                if mutation=='metrics' and name=='eval1.json':value['metrics']={'AP':.9}
                if mutation=='baseline' and name=='shared-baseline.json':value['metrics']={'AP':.8}
                return value
            def verify(path,context):
                epoch=int(str(path)[-1]);return {'position':{'epoch':epoch,'next_batch':0,'global_step':0 if mutation=='position' else epoch*221}}
            with (self.subTest(mutation=mutation),patch.object(replay,'strict_load',side_effect=read),
                  patch.object(replay,'checked_file',side_effect=lambda e:Path(e['path'])),
                  patch.object(replay,'CocoTrainingDataset',return_value=Dataset()),
                  patch.object(replay,'verify_checkpoint',side_effect=verify),
                  patch.object(replay,'digest',return_value='rights'),
                  patch.object(replay,'saved_rows',return_value={42:[]})):
                if mutation:
                    with self.assertRaises(ValueError):replay.load_saved_review('manifest','x',None,'rights')
                else:
                    ds,models,record=replay.load_saved_review('manifest','x',None,'rights')
                    self.assertEqual(set(models),{'pretrained','epoch-01','epoch-02'})
                    self.assertEqual(record['saved_configuration'],report['contract'])
                    self.assertEqual(record['provenance']['new_inference_calls'],0)

    def test_artifact_hash_and_workspace_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'value.json';path.write_text('{}')
            entry={'path':str(path),'sha256':replay.digest(path)}
            with patch.object(replay,'ROOT',Path(tmp)):
                self.assertEqual(replay.checked_file(entry),path.resolve())
                path.write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError,'hash mismatch'):replay.checked_file(entry)
            entry['sha256']=replay.digest(path)
            with self.assertRaisesRegex(ValueError,'path/hash mismatch'):replay.checked_file(entry)

    def test_cli_explicit_arguments_required_and_no_implicit_gui(self):
        for argv in (['--saved-review','anything'],['--run-choice','x'],['--image-id','42']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):main(argv)
        with tempfile.TemporaryDirectory() as tmp:
            record={'saved_run_choice':'x'}
            with (patch('scripts.taco_visual_review.ROOT',Path(tmp)),
                  patch.object(replay,'load_saved_review',return_value=(None,{},record)),
                  patch('scripts.taco_visual_review.show_review') as show,
                  contextlib.redirect_stdout(io.StringIO())):
                (Path(tmp)/'runs/training').mkdir(parents=True)
                self.assertEqual(main(['--saved-review','manifest.json','--run-choice','x']),0)
                show.assert_not_called()

    def test_schema_and_corpus_reject_before_checkpoint_or_photos(self):
        with patch.object(replay,'strict_load',return_value={}), self.assertRaisesRegex(ValueError,'schema|Unsupported'):
            replay.load_saved_review(None,'x',None,None)
        manifest={'schema_version':'bantai.cross-seed-review.v1','runs':{'x':{}},'dataset':{'expected':True}}
        ds=SimpleNamespace(context={},data={'annotations':[]})
        with (patch.object(replay,'strict_load',return_value=manifest),
              patch.object(replay,'CocoTrainingDataset',return_value=ds),
              patch.object(replay,'verify_checkpoint',side_effect=AssertionError('No checkpoint reads')),
              self.assertRaisesRegex(ValueError,'corpus mismatch')):
            replay.load_saved_review(None,'x',None,None)


if __name__=='__main__':unittest.main()
