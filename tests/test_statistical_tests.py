import unittest
import numpy as np
import pandas as pd
from core.statistical_tests import marginal_test, conditional_test, robust_orientation_test

class TestStatisticalTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
    
    def test_marginal_test_different_distributions(self):
        data_obs = pd.DataFrame({'Vk': np.random.binomial(1, 0.5, 100)})
        data_exp = pd.DataFrame({'Vk': np.random.binomial(1, 0.1, 100)})
        self.assertTrue(marginal_test('Vk', data_obs, data_exp, 0.05))
    
    def test_marginal_test_same_distributions(self):
        data_obs = pd.DataFrame({'Vk': np.random.binomial(1, 0.5, 100)})
        data_exp = pd.DataFrame({'Vk': np.random.binomial(1, 0.5, 100)})
        self.assertFalse(marginal_test('Vk', data_obs, data_exp, 0.05))
    
    def test_marginal_test_small_sample(self):
        data_obs = pd.DataFrame({'Vk': np.random.binomial(1, 0.5, 40)})
        data_exp = pd.DataFrame({'Vk': np.random.binomial(1, 0.1, 40)})
        self.assertFalse(marginal_test('Vk', data_obs, data_exp, 0.05))
    
    def test_marginal_test_multiclass(self):
        data_obs = pd.DataFrame({'Vk': np.random.choice([0, 1, 2], 100)})
        data_exp = pd.DataFrame({'Vk': np.random.choice([0, 1, 2], 100, p=[0.8, 0.1, 0.1])})
        self.assertTrue(marginal_test('Vk', data_obs, data_exp, 0.05))
    
    def test_marginal_test_invalid_chi2(self):
        data_obs = pd.DataFrame({'Vk': [0]*95 + [1]*5})
        data_exp = pd.DataFrame({'Vk': [0]*99 + [1]*1})
        self.assertFalse(marginal_test('Vk', data_obs, data_exp, 0.05))
    
    def test_conditional_test_conditional_differs(self):
        obs_Vi = np.random.binomial(1, 0.5, 1000)
        obs_Vk = (obs_Vi * 3).astype(int)
        
        exp_Vi = np.random.binomial(1, 0.9, 1000)
        exp_Vk = (exp_Vi * 2).astype(int)
        
        data_obs = pd.DataFrame({'Vi': obs_Vi, 'Vk': obs_Vk})
        data_exp = pd.DataFrame({'Vi': exp_Vi, 'Vk': exp_Vk})
        B = ['Vi']
        self.assertTrue(conditional_test('Vk', B, data_obs, data_exp, 0.01, 5))
    
    def test_conditional_test_conditional_same(self):
        obs_Vi = np.random.binomial(1, 0.5, 1000)
        obs_Vk = (obs_Vi * 3).astype(int)
        
        exp_Vi = np.random.binomial(1, 0.9, 1000)
        exp_Vk = (exp_Vi * 3).astype(int)
        
        data_obs = pd.DataFrame({'Vi': obs_Vi, 'Vk': obs_Vk})
        data_exp = pd.DataFrame({'Vi': exp_Vi, 'Vk': exp_Vk})
        B = ['Vi']
        self.assertFalse(conditional_test('Vk', B, data_obs, data_exp, 0.01, 5))
    
    def test_conditional_test_empty_B(self):
        obs_Vi = np.random.binomial(1, 0.5, 1000)
        obs_Vk = (obs_Vi * 3).astype(int)
        
        exp_Vi = np.random.binomial(1, 0.9, 1000)
        exp_Vk = (exp_Vi * 2).astype(int)
        
        data_obs = pd.DataFrame({'Vi': obs_Vi, 'Vk': obs_Vk})
        data_exp = pd.DataFrame({'Vi': exp_Vi, 'Vk': exp_Vk})
        self.assertFalse(conditional_test('Vk', [], data_obs, data_exp, 0.01, 5))
    
    def test_conditional_test_small_sample(self):
        obs_Vi = np.random.binomial(1, 0.5, 40)
        obs_Vk = (obs_Vi * 3).astype(int)
        
        exp_Vi = np.random.binomial(1, 0.9, 40)
        exp_Vk = (exp_Vi * 2).astype(int)
        
        data_obs = pd.DataFrame({'Vi': obs_Vi, 'Vk': obs_Vk})
        data_exp = pd.DataFrame({'Vi': exp_Vi, 'Vk': exp_Vk})
        B = ['Vi']
        self.assertFalse(conditional_test('Vk', B, data_obs, data_exp, 0.01, 5))
    
    def test_robust_orientation_test_neither(self):
        obs_Vi = np.random.binomial(1, 0.5, 1000)
        obs_Vk = np.random.binomial(1, 0.3, 1000)
        
        exp_Vi = np.random.binomial(1, 0.9, 1000)
        exp_Vk = np.random.binomial(1, 0.3, 1000)
        
        data_obs = pd.DataFrame({'Vi': obs_Vi, 'Vk': obs_Vk})
        data_exp = pd.DataFrame({'Vi': exp_Vi, 'Vk': exp_Vk})
        B = ['Vi']

        result = robust_orientation_test('Vi', 'Vk', B, data_obs, data_exp, 0.05, 0.01)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main(verbosity=2)