"""Generated trajectory fixtures; no images, model, backward or optimizer."""
import unittest
import importlib.util
from scripts.configuration import ROOT
support_path=ROOT/'work/scale576_trajectory/support.py'
if not support_path.is_file():
    raise unittest.SkipTest('Optional local D-058 audit helpers are not published; work/ remains private')
spec=importlib.util.spec_from_file_location('trajectory_support',support_path)
support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)

class TrajectoryTests(unittest.TestCase):
    def test_exact_seven_step_ordering_rejects_missing_duplicate_and_extra(self):
        self.assertEqual(support.ordered_steps([442,0,350,221,400,250,300]),support.STEPS)
        for steps in ((0,221,250,300,350,400),(0,221,250,300,350,400,400),support.STEPS+(450,)):
            with self.assertRaises(ValueError):support.ordered_steps(steps)

    def test_exact_periodic_and_epoch_checkpoint_positions(self):
        for step,nb in ((250,29),(300,79),(350,129),(400,179)):
            expected={'epoch':1,'next_batch':nb,'global_step':step}
            self.assertEqual(support.expected_position(step),expected)
            self.assertEqual(support.verify_position(expected,step),expected)
            for key in expected:
                bad=dict(expected);bad[key]+=1
                with self.assertRaises(ValueError):support.verify_position(bad,step)
        self.assertEqual(support.expected_position(221),{'epoch':1,'next_batch':0,'global_step':221})
        self.assertEqual(support.expected_position(442),{'epoch':2,'next_batch':0,'global_step':442})
        with self.assertRaises(ValueError):support.expected_position(0)

    def test_locked_denominators_transitions_and_reversals_preserve_gt_identity(self):
        widths=[3]*14+[6]*10+[12]*15+[20]*36
        objects=[{'annotation_id':i,'resized_bbox_width':w,'resized_bbox_height':w} for i,w in enumerate(widths)]
        missing={'any_class_coverage':False,'correct_class_coverage':False}
        wrong={'any_class_coverage':True,'correct_class_coverage':False}
        correct={'any_class_coverage':True,'correct_class_coverage':True}
        values={s:{i:dict(missing) for i in range(75)} for s in support.STEPS}
        for s in (221,250,300,350,400,442):values[s][40]=dict(correct if s in (221,300) else wrong)
        for s in (250,300,350,400,442):values[s][0]=dict(correct)
        # Pixel growth at later steps must not change the original bin.
        values[442][0]['resized_bbox_width']=99
        result=support.track(objects,values)
        self.assertEqual(result['E1_E2_changes'],{'gains':[0],'losses':[40]})
        for p in result['points'].values():
            self.assertEqual([p['bins'][g]['N'] for g in support.GROUPS],[14,10,15,36])
            self.assertEqual(sum(p['states'].values()),75)
        self.assertEqual(result['state_trajectories'][40],[support.STATES[2],support.STATES[0],support.STATES[1],support.STATES[0],support.STATES[1],support.STATES[1],support.STATES[1]])
        for edge in result['transitions']:self.assertEqual(sum(sum(row.values()) for row in edge['matrix_counts'].values()),75)
        self.assertEqual(result['transitions'][1]['matrix_ids']['correct_class']['wrong_class_only'],[40])
        self.assertEqual(result['transitions'][2]['matrix_ids']['wrong_class_only']['correct_class'],[40])
        values[300].pop(0)
        with self.assertRaises(ValueError):support.track(objects,values)
