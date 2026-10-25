"""Saved integration replay fixtures, with all artifact reads mocked."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts import taco_saved_review as replay
from scripts.taco_weight_integration import RECIPES


class IntegratedReviewTests(unittest.TestCase):
    def test_three_explicit_panels_and_audit_tamper_rejection(self):
        ev={'metrics':{'AP':.2}}
        contract={'fixture':True}
        state='blend_350_442_5050'
        refs={str(s):{'path':f'ref{s}.json','sha256':str(s)} for s in (350,442)}
        audit={'authorization':'D-061','status':'completed','contract':contract,'protocol':replay.PROTOCOL,
            'new_model_forwards':188,'optimizer_updates':0,'backward_passes':0,'test_split_access':0,
            'training_photo_reads':0,'reference_GPU_forwards':0,'checkpoint_bundles_before':50,
            'checkpoint_bundles_after':50,'protected_after':{'unchanged':True},'references':refs,
            'weight_checks':{'trainable_parameter_tensors':558,'all_frozen_parameters_and_buffers_bitwise_identical':True},
            'states':{n:{'weights':{str(k):v for k,v in w.items()},'evaluation_file':'best.json',
                        'evaluation_sha256':'best','forwards':47,'unchanged_after_evaluation':True,
                        'metrics':ev['metrics']} for n,w in RECIPES.items()}}
        entry={'integration':{'path':'audit.json'},'panel_labels':{'pretrained':'Step350','epoch-01':'Step442','epoch-02':state},
            'panels':{'pretrained':{'evaluation':refs['350'],'reference_step':350},
                      'epoch-01':{'evaluation':refs['442'],'reference_step':442},
                      'epoch-02':{'evaluation':{'path':'best.json','sha256':'best'},'integration_state':state}}}
        ds=SimpleNamespace(images=[{'id':777}],data={})
        for mutation in (None,'forwards','frozen','metrics','weights','source'):
            value=copy.deepcopy(audit);saved_entry=copy.deepcopy(entry)
            if mutation=='forwards':value['new_model_forwards']=189
            if mutation=='frozen':value['weight_checks']['all_frozen_parameters_and_buffers_bitwise_identical']=False
            if mutation=='metrics':value['states'][state]['metrics']={}
            if mutation=='weights':value['states'][state]['weights']={'350':.7,'442':.3}
            if mutation=='source':saved_entry['panels']['pretrained']['reference_step']=442
            with (self.subTest(mutation=mutation),
                  patch.object(replay,'checked_file',side_effect=lambda a:Path(a['path'])),
                  patch.object(replay,'strict_load',side_effect=lambda p:value if str(p)=='audit.json' else ev),
                  patch.object(replay,'saved_rows',return_value={777:[]}),patch.object(replay,'digest',return_value='manifest')):
                if mutation:
                    with self.assertRaises(ValueError):replay.load_integrated_review('manifest','choice',{'images':[{'image_id':777}]},saved_entry,ds,{'contract':contract})
                else:
                    _,models,record=replay.load_integrated_review('manifest','choice',{'images':[{'image_id':777}]},saved_entry,ds,{'contract':contract})
                    self.assertEqual(set(models),{'pretrained','epoch-01','epoch-02'})
                    self.assertEqual(record['saved_configuration']['replay_panels']['pretrained'],'Step350')
                    self.assertEqual(record['provenance']['new_inference_calls'],0)
