import unittest
from drvpinn import get_params, parser

class TestClArgs(unittest.TestCase):


    def test_custom_file(self):
        args = ['--params', 'test/test_params.ini']
        parsed_args = parser.parse_args(args)
        params = get_params(parsed_args)
        self.assertEqual(params.tag, 'tmp')
        self.assertEqual(params.epochs, 1)
        self.assertEqual(params.layers, 1)
        self.assertEqual(params.neurons_per_layer, 1)
        self.assertEqual(params.learning_rate, 0.1)
        self.assertEqual(params.use_best_pinn, True)
        self.assertEqual(params.equation, 'delta')
        self.assertEqual(params.eps, 0.1)
        self.assertEqual(params.Xd, 0.1)
        self.assertEqual(params.compute_error, False)
        self.assertEqual(params.n_points_x, 1)
        self.assertEqual(params.n_points_error, 1)
        self.assertEqual(params.n_test_func, 1)
        self.assertEqual(params.integration_rule_loss, 'trapz')
        self.assertEqual(params.integration_rule_norm, 'trapz')
        self.assertEqual(params.integration_rule_error, 'trapz')
        self.assertEqual(params.atol, 0.1)
        self.assertEqual(params.rtol, 0.1)

    def test_cl_args(self):
        args = ['--tag', 'my_tag', 
                '--epochs', '42',
                '--layers', '42',
                '--neurons-per-layer', '42',
                '--learning-rate', '0.42',
                '--no-use-best-pinn',
                '--equation', 'ad',
                '--eps', '0.42',
                '--Xd', '0.42',
                '--compute-error',
                '--n-points-x', '42',
                '--n-points-error', '42',
                '--n-test-func', '42',
                '--integration-rule-loss', 'midpoint',
                '--integration-rule-norm', 'midpoint',  
                '--integration-rule-error', 'midpoint',  
                '--atol', '0.42',
                '--rtol', '0.42',
                '--params', 'test/test_params.ini']
        parsed_args = parser.parse_args(args)
        params = get_params(parsed_args)
        self.assertEqual(params.tag, 'my_tag')
        self.assertEqual(params.epochs, 42)
        self.assertEqual(params.layers, 42)
        self.assertEqual(params.neurons_per_layer, 42)
        self.assertEqual(params.learning_rate, 0.42)
        self.assertEqual(params.use_best_pinn, False)
        self.assertEqual(params.equation, 'ad')
        self.assertEqual(params.eps, 0.42)
        self.assertEqual(params.Xd, 0.42)
        self.assertEqual(params.compute_error, True)
        self.assertEqual(params.n_points_x, 42)
        self.assertEqual(params.n_points_error, 42)
        self.assertEqual(params.n_test_func, 42)
        self.assertEqual(params.integration_rule_loss, 'midpoint')
        self.assertEqual(params.integration_rule_norm, 'midpoint')
        self.assertEqual(params.integration_rule_error, 'midpoint')
        self.assertEqual(params.atol, 0.42)
        self.assertEqual(params.rtol, 0.42)

if __name__ == '__main__':
    unittest.main()