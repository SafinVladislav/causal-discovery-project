import unittest
import pandas as pd
import numpy as np
from core.statistical_tests import marginal_test, conditional_test, robust_orientation_test

np.random.seed(0)

class TestRobustOrientationFunctions(unittest.TestCase):

    def test_marginal_small_sample_returns_none(self):
        # fewer than 50 observations in either dataset => None
        df1 = pd.DataFrame({'Y': np.random.choice(['A', 'B'], size=10)})
        df2 = pd.DataFrame({'Y': np.random.choice(['A', 'B'], size=10)})
        self.assertIsNone(marginal_test('Y', df1, df2))

    def test_marginal_significant_and_non_significant(self):
        # create two datasets with clear difference in marginal distribution
        obs = pd.DataFrame({'Y': ['A'] * 40 + ['B'] * 10})
        exp = pd.DataFrame({'Y': ['A'] * 10 + ['B'] * 40})
        self.assertTrue(marginal_test('Y', obs, exp, alpha=0.05))

        # and two datasets that are similar => not significant
        obs2 = pd.DataFrame({'Y': ['A'] * 25 + ['B'] * 25})
        exp2 = pd.DataFrame({'Y': ['A'] * 25 + ['B'] * 25})
        self.assertFalse(marginal_test('Y', obs2, exp2, alpha=0.05))

    def test_conditional_empty_B_and_small_sample(self):
        # empty B => should return False
        obs = pd.DataFrame({'Y': np.random.binomial(1, 0.5, size=60)})
        exp = pd.DataFrame({'Y': np.random.binomial(1, 0.5, size=60)})
        self.assertFalse(conditional_test('Y', [], obs, exp))

        # small sample => False
        obs_small = pd.DataFrame({'Y': np.random.binomial(1, 0.5, size=10)})
        exp_small = pd.DataFrame({'Y': np.random.binomial(1, 0.5, size=10)})
        self.assertFalse(conditional_test('Y', ['X'], obs_small, exp_small))

    def test_conditional_binary_detects_difference(self):
        # Create a binary outcome where the dataset_group has a strong effect
        n = 80
        X_obs = np.random.normal(loc=0.0, scale=1.0, size=n)
        X_exp = np.random.normal(loc=0.0, scale=1.0, size=n)

        # make probability very different between groups to ensure significance
        p_obs = 0.8
        p_exp = 0.2
        Y_obs = np.random.binomial(1, p_obs, size=n)
        Y_exp = np.random.binomial(1, p_exp, size=n)

        obs = pd.DataFrame({'Y': Y_obs, 'X': X_obs})
        exp = pd.DataFrame({'Y': Y_exp, 'X': X_exp})

        self.assertTrue(conditional_test('Y', ['X'], obs, exp, alpha=0.05))

    def test_conditional_binary_no_difference_returns_false(self):
        # Create data where groups are similar once conditioning on X (no dataset effect)
        n = 80
        X_obs = np.random.normal(size=n)
        X_exp = np.random.normal(size=n)

        # probability depends on X, but is the same function in both datasets
        logits = lambda x: 1 / (1 + np.exp(-x))
        p_obs = logits(X_obs)
        p_exp = logits(X_exp)
        Y_obs = np.random.binomial(1, p_obs)
        Y_exp = np.random.binomial(1, p_exp)

        obs = pd.DataFrame({'Y': Y_obs, 'X': X_obs})
        exp = pd.DataFrame({'Y': Y_exp, 'X': X_exp})

        self.assertFalse(conditional_test('Y', ['X'], obs, exp, alpha=0.05))

    def test_robust_orientation_various_scenarios(self):
        # 1) marginal True, conditional False -> returns (Vi, Vk)
        obs = pd.DataFrame({'Y': ['A'] * 40 + ['B'] * 10})
        exp = pd.DataFrame({'Y': ['A'] * 10 + ['B'] * 40})
        # provide empty B so conditional_test returns False
        res = robust_orientation_test('X', 'Y', [], obs, exp)
        self.assertEqual(res, ('X', 'Y'))

        # 2) conditional True, marginal False -> returns (Vk, Vi)
        # construct conditional difference with same marginals
        n = 60
        # B is binary with equal frequency
        B_obs = np.array([0] * (n // 2) + [1] * (n // 2))
        B_exp = np.array([0] * (n // 2) + [1] * (n // 2))
        # In obs Y == B, in exp Y == 1 - B. Marginal distribution of Y will be 50/50 in both.
        Y_obs = B_obs.copy()
        Y_exp = 1 - B_exp

        obs_df = pd.DataFrame({'Y': Y_obs, 'B1': B_obs})
        exp_df = pd.DataFrame({'Y': Y_exp, 'B1': B_exp})

        # marginal should be non-significant but conditional should be significant
        marg = marginal_test('Y', obs_df, exp_df)
        cond = conditional_test('Y', ['B1'], obs_df, exp_df)
        self.assertFalse(marg)
        self.assertTrue(cond)

        res2 = robust_orientation_test('I', 'Y', ['B1'], obs_df, exp_df)
        self.assertEqual(res2, ('Y', 'I'))

        # 3) both False -> returns None
        obs_same = pd.DataFrame({'Y': np.random.choice(['A', 'B'], size=60)})
        exp_same = pd.DataFrame({'Y': np.random.choice(['A', 'B'], size=60)})
        res3 = robust_orientation_test('I', 'Y', [], obs_same, exp_same)
        self.assertIsNone(res3)

        # 4) both True -> returns None (ambiguous orientation)
        # Use a strong difference that will trigger both marginal and conditional tests
        n = 80
        obs = pd.DataFrame({'Y': ['A'] * 60 + ['B'] * 20, 'Z': np.random.normal(size=n)})
        exp = pd.DataFrame({'Y': ['A'] * 20 + ['B'] * 60, 'Z': np.random.normal(size=n)})
        res4 = robust_orientation_test('I', 'Y', ['Z'], obs, exp)
        # ambiguous: both tests should detect differences -> result is None
        self.assertIsNone(res4)


if __name__ == '__main__':
    unittest.main()
