"""Software fixtures for D-073; no dataset photos, model or GPU."""
import copy
import unittest
from unittest.mock import patch
from scripts.taco_stage5_data_expansion_audit import proposal_audit, locked_duplicate_members, summarize
from scripts.taco_preparation import make_plan, load_policy


def fixture(names):
    labels=[{'id':i,'name':str(i),'taco_categories':['base'+str(i)]} for i in range(9)]
    rows=[]
    for i,name in enumerate(names):
        rows.append({'index':i,'original':{'id':i,'image_id':1,'category_id':i,'bbox':[0,0,2,2],'area':4,'iscrowd':0},
          'original_category':{'id':i,'name':name},'issues':[],
          'bantai_label':'0' if name=='base0' else None,'bantai_label_id':0 if name=='base0' else None,
          'handling':'unlabeled_requires_decision' if name=='Unlabeled litter' else 'excluded_requires_decision'})
    return {'taxonomy':{'labels':labels},'annotations':rows,
      'images':[{'index':0,'original':{'id':1,'file_name':'fixture.png','width':10,'height':10},
                 'status':'readable','issues':[],'exif_orientation':1}],
      'summary':{'source':{'dataset_root':'fixture'}},'source_document':{}}


def plan(audit):
    fp={'file_sha256':'a','pixel_sha256':'b','dhash64':'0'*16,'ahash64':'0'*16,'aspect_ratio':1.}
    with patch('scripts.taco_preparation.fingerprint',return_value=fp):return make_plan(audit,load_policy())


class ExpansionAuditTests(unittest.TestCase):
    def test_strict_whole_image_and_accessory_only_relaxation(self):
        original=fixture(['base0','Plastic bottle cap'])
        self.assertFalse(plan(original)['eligibility'][0]['eligible'])
        proposed,omitted=proposal_audit(original,True,True)
        self.assertTrue(plan(proposed)['eligibility'][0]['eligible'])
        self.assertEqual(omitted,[1]);self.assertEqual(len(original['annotations']),2)

    def test_unlabeled_and_unapproved_exclusions_remain_quarantined(self):
        for name in ('Unlabeled litter','Broken glass','Battery','Tupperware'):
            proposed,_=proposal_audit(fixture(['base0',name]),True,True)
            self.assertFalse(plan(proposed)['eligibility'][0]['eligible'])

    def test_invalid_crowd_and_ignore_accessories_cannot_be_dropped(self):
        for key,value in [('area',0),('iscrowd',1),('ignore',1)]:
            original=fixture(['base0','Plastic lid']);original['annotations'][1]['original'][key]=value
            proposed,omitted=proposal_audit(original,True,True)
            self.assertEqual(omitted,[])
            self.assertFalse(plan(proposed)['eligibility'][0]['eligible'])

    def test_sibling_mapping_and_other_plastic_sensitivity(self):
        expected={'Polypropylene bag':2,'Six pack rings':3,'Food Can':5,'Aerosol':5,
                  'Disposable food container':6,'Spread tub':6,'Other plastic container':6,
                  'Foam cup':7,'Foam food container':7,'Plastified paper bag':8}
        for name,cid in expected.items():
            original=fixture([name]);proposed,_=proposal_audit(original,True)
            self.assertEqual(proposed['annotations'][0]['bantai_label_id'],cid)
            self.assertTrue(plan(proposed)['eligibility'][0]['eligible'])
            self.assertIsNone(original['annotations'][0]['bantai_label'])
        base=fixture(['Other plastic'])
        self.assertIsNone(proposal_audit(base,True)[0]['annotations'][0]['bantai_label'])
        self.assertEqual(proposal_audit(base,True,False,True)[0]['annotations'][0]['bantai_label_id'],3)

    def test_empty_after_filtering_is_quarantined(self):
        proposed,_=proposal_audit(fixture(['Glass bottle']),True,True)
        self.assertFalse(plan(proposed)['eligibility'][0]['eligible'])

    def test_duplicate_leakage_components_are_transitive(self):
        def fp(n):return {'file_sha256':str(n),'pixel_sha256':str(n),'dhash64':f'{n:x}','ahash64':f'{n:x}','aspect_ratio':1.}
        # 0 -> 1 -> 3 are near-neighbor links; 0 and 3 differ by two bits.
        blocked,evidence=locked_duplicate_members({0:fp(0),1:fp(1),2:fp(3),3:fp(255)}, {0},
             {'max_dhash_distance':1,'max_ahash_distance':1,'max_relative_aspect_difference':.02})
        self.assertEqual(blocked,{0,1,2});self.assertTrue(evidence['near_pairs'])

    def test_new_unlabeled_exact_duplicate_quarantines_existing_clean_image(self):
        audit=fixture(['base0','Unlabeled litter'])
        audit['images'].append(copy.deepcopy(audit['images'][0]))
        audit['images'][1]['original']['id']=2
        audit['images'][1]['index']=1
        audit['annotations'][1]['original']['image_id']=2
        rows=plan(audit)['eligibility']
        self.assertEqual(rows[0]['reasons'],['exact_duplicate_quarantine'])
        self.assertFalse(rows[0]['eligible'])
        self.assertIn('unlabeled_annotation',rows[1]['reasons'])

    def test_frozen_and_blocked_ids_never_count_as_training(self):
        audit=fixture(['base0']);p=plan(audit)
        for locked,blocked in [({1},set()),(set(),{1})]:
            result=summarize(p,audit,locked,blocked,set(),[])
            self.assertEqual(result['training_images'],0);self.assertEqual(result['boxes'],0)


if __name__=='__main__':unittest.main()
