import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, chi2, ttest_rel
from statsmodels.genmod.families import Binomial
import warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import norm

def marginal_test(Vk, data_obs, data_exp, alpha):
    obs_counts = data_obs[Vk].value_counts().sort_index()
    exp_counts = data_exp[Vk].value_counts().sort_index()

    all_categories = sorted(set(obs_counts.index) | set(exp_counts.index))
    obs_counts = obs_counts.reindex(all_categories, fill_value=0)
    exp_counts = exp_counts.reindex(all_categories, fill_value=0)

    if obs_counts.sum() < 50 or exp_counts.sum() < 50:
        return False

    contingency_table = np.array([obs_counts.values, exp_counts.values])
    _, p_value, _, expected_counts = chi2_contingency(contingency_table)

    low_expected_count_percentage = np.mean(expected_counts < 5) * 100
    is_chi2_valid = (low_expected_count_percentage <= 20) and (np.all(expected_counts >= 1))

    return False if not is_chi2_valid else p_value < alpha

def should_use_separate_models(mixed_scores, obs_scores, exp_scores, n_obs, n_exp, alpha):
    # Check if scores are effectively identical across models
    if np.allclose(np.array(mixed_scores), np.array(obs_scores)) and np.allclose(np.array(mixed_scores), np.array(exp_scores)):
        print("→ IDENTICAL SCORES. Use mixed model.")
        return False
    
    mean_mixed = np.mean(mixed_scores)
    mean_obs = np.mean(obs_scores)
    mean_exp = np.mean(exp_scores)
    
    N = n_obs + n_exp
    w_obs = n_obs / N
    w_exp = n_exp / N
    weighted_separate = w_obs * mean_obs + w_exp * mean_exp
    
    d = mean_mixed - weighted_separate
    
    print(f"→ Difference (mixed - weighted separate): {d:.5f}")
    
    if d <= 0:
        p_value = 1.0
    else:
        # Estimate variances of the means
        if len(mixed_scores) < 2 or len(obs_scores) < 2 or len(exp_scores) < 2:
            return False  # Not enough splits for variance estimate
        
        var_mixed = np.var(mixed_scores, ddof=1) / len(mixed_scores)
        var_obs = np.var(obs_scores, ddof=1) / len(obs_scores)
        var_exp = np.var(exp_scores, ddof=1) / len(exp_scores)
        
        var_weighted = w_obs**2 * var_obs + w_exp**2 * var_exp
        
        se_d = np.sqrt(var_mixed + var_weighted)  # Conservative estimate
        
        if se_d == 0:
            return d > 0  # Rare edge case
        
        z = d / se_d
        p_value = norm.sf(z)  # One-tailed p-value for d > 0
    
    print(f"→ p-value: {p_value:.2e}")
    
    return p_value < alpha

def compute_cv_scores(data, Vk, B, categorical_features_B, n_splits, lgbm_params_base):
    y, class_names = pd.factorize(data[Vk])
    target_cardinality = len(class_names)
    if target_cardinality < 2:
        return None

    lgbm_params = lgbm_params_base.copy()
    if target_cardinality == 2:
        lgbm_params['objective'] = 'binary'
    else:
        lgbm_params['objective'] = 'multiclass'
        lgbm_params['num_class'] = target_cardinality

    for col in categorical_features_B:
        data[col] = data[col].astype('category')

    X = data[B]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    cv_scores = []

    for train_idx, test_idx in skf.split(data, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(
            X_train, y_train,
            categorical_feature=categorical_features_B
        )
        pred_proba = model.predict_proba(X_test)

        loss = log_loss(y_test, pred_proba, labels=np.arange(target_cardinality))
        cv_scores.append(loss)

    return cv_scores

def conditional_test(Vk, B, data_obs, data_exp, alpha, n_splits):
    if not B:
        return False
    if len(data_obs) < 50 or len(data_exp) < 50:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        try:
            combined_data = pd.concat([data_obs, data_exp]).copy()

            categorical_features_B = [
                col for col in B 
                if combined_data[col].dtype in ['object', 'category', 'bool']
            ]

            lgbm_params_base = {
                'n_estimators': 30,
                'learning_rate': 0.1,
                'verbose': -1,
                'n_jobs': -1
            }

            # Compute for mixed (combined)
            mixed_scores = compute_cv_scores(combined_data, Vk, B, categorical_features_B, n_splits, lgbm_params_base)
            if mixed_scores is None:
                return False
            print("Mixed model scores:")
            for score in mixed_scores:
                print(score)

            # Compute for obs
            obs_scores = compute_cv_scores(data_obs, Vk, B, categorical_features_B, n_splits, lgbm_params_base)
            if obs_scores is None:
                return False
            print("Obs model scores:")
            for score in obs_scores:
                print(score)

            # Compute for exp
            exp_scores = compute_cv_scores(data_exp, Vk, B, categorical_features_B, n_splits, lgbm_params_base)
            if exp_scores is None:
                return False
            print("Exp model scores:")
            for score in exp_scores:
                print(score)

            if len(mixed_scores) < 2:
                return False

            separate_scores = obs_scores + exp_scores

            # For logging weighted means (informational)
            n_obs, n_exp = len(data_obs), len(data_exp)
            mean_mixed = np.mean(mixed_scores)
            mean_obs = np.mean(obs_scores)
            mean_exp = np.mean(exp_scores)
            weighted_separate = (n_obs * mean_obs + n_exp * mean_exp) / (n_obs + n_exp)
            print(f"→ Mean mixed: {mean_mixed:.5f}, Weighted mean separate: {weighted_separate:.5f}")

            return should_use_separate_models(mixed_scores, obs_scores, exp_scores, n_obs, n_exp, alpha)

        except (np.linalg.LinAlgError, ValueError, IndexError):
            return False

N_SPLITS = 10

def robust_orientation_test(Vi, Vk, B, data_obs, data_exp, alpha1, alpha2):
    marg = False#marginal_test(Vk, data_obs, data_exp, alpha1)
    cond = conditional_test(Vk, B, data_obs, data_exp, alpha2, N_SPLITS)
    
    if marg and not cond:
        return (Vi, Vk)
    elif cond and not marg:
        return (Vk, Vi)
    else:
        return None