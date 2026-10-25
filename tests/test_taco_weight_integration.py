"""Small CPU tensor fixtures; no detector, photo, update or GPU execution."""
import unittest
import torch
from scripts.taco_weight_integration import RECIPES,average_trainable,bitwise_equal,canonical_saved_keys


class WeightIntegrationTests(unittest.TestCase):
    def test_saved_aliases_preserve_tied_parameter_identity(self):
        model=torch.nn.Module();model.a=torch.nn.Linear(2,2);model.b=model.a
        model.register_buffer('runtime_only',torch.ones(1),persistent=False)
        mapping=canonical_saved_keys(model,['b.weight','b.bias'])
        self.assertEqual(mapping,{'b.weight':'a.weight','b.bias':'a.bias'})
        for names in (['b.weight'],['b.weight','a.weight','b.bias'],['wrong']):
            with self.assertRaises(ValueError):canonical_saved_keys(model,names)

    def test_exact_recipes_and_unchanged_sources(self):
        states={s:{'weight':torch.tensor([float(i),float(i+3)])} for i,s in enumerate((221,350,442))}
        copies={s:v['weight'].clone() for s,v in states.items()}
        expected=[1.5,1.75,1.,1.]
        for (name,weights),value in zip(RECIPES.items(),expected):
            result=average_trainable(states,weights)['weight']
            torch.testing.assert_close(result,torch.tensor([value,value+3]))
            self.assertTrue(all(bitwise_equal(states[s]['weight'],copies[s]) for s in states))
            self.assertNotEqual(result.data_ptr(),states[221]['weight'].data_ptr())

    def test_reject_mismatch_nonfinite_and_invalid_weights(self):
        base={221:{'x':torch.ones(2)},442:{'x':torch.ones(2)}}
        for weights in ({221:.2,442:.2},{221:-1.,442:2.},{221:float('nan')}):
            with self.assertRaises(ValueError):average_trainable(base,weights)
        for bad in ({'y':torch.ones(2)},{'x':torch.ones(3)},
                    {'x':torch.ones(2,dtype=torch.float64)},{'x':torch.tensor([float('nan'),0.])}):
            with self.assertRaises(ValueError):average_trainable({221:base[221],442:bad},{221:.5,442:.5})

    def test_bitwise_guard_distinguishes_signed_zero(self):
        self.assertFalse(bitwise_equal(torch.tensor([0.]),torch.tensor([-0.])))
