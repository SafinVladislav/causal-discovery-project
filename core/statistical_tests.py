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

"""def should_keep_full_model(full_scores, reduced_scores, alpha):
    if np.allclose(np.array(full_scores) - np.array(reduced_scores), 0):
        print("→ IDENTICAL SCORES. Keep reduced model.")
        return True
    
    t_stat, p_value = ttest_rel(full_scores, reduced_scores, alternative='less')
    
    print(f"→ p-value: {p_value:.2e}")
    
    return p_value < alpha

def conditional_test(Vk, B, data_obs, data_exp, alpha, n_splits):
    if not B:
        return False
    if len(data_obs) < 50 or len(data_exp) < 50:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        try:
            combined_data = pd.concat([
                data_obs.assign(dataset_group=0),
                data_exp.assign(dataset_group=1)
            ]).copy()

            y, class_names = pd.factorize(combined_data[Vk])

            target_cardinality = len(class_names)
            if target_cardinality < 2:
                return False

            categorical_features_B = [
                col for col in B 
                if combined_data[col].dtype in ['object', 'category', 'bool']
            ]
            
            features_reduced = B
            features_full = B + ['dataset_group']
            
            cat_features_reduced = categorical_features_B
            cat_features_full = categorical_features_B + ['dataset_group']

            for col in cat_features_full:
                combined_data[col] = combined_data[col].astype('category')

            X_reduced = combined_data[features_reduced]
            X_full = combined_data[features_full]

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            
            reduced_model_scores = []
            full_model_scores = []

            lgbm_params = {
                'objective': 'binary' if target_cardinality == 2 else 'multiclass',
                'num_class': target_cardinality if target_cardinality > 2 else 1,
                'n_estimators': 30,
                'learning_rate': 0.1,
                'verbose': -1,
                'n_jobs': -1,
                'class_weight': 'balanced'
            }

            for train_idx, test_idx in skf.split(combined_data, y):
                X_reduced_train, X_reduced_test = X_reduced.iloc[train_idx], X_reduced.iloc[test_idx]
                X_full_train, X_full_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model_reduced = lgb.LGBMClassifier(**lgbm_params)
                model_reduced.fit(
                    X_reduced_train, y_train,
                    categorical_feature=cat_features_reduced
                )
                pred_proba_reduced = model_reduced.predict_proba(X_reduced_test)

                model_full = lgb.LGBMClassifier(**lgbm_params)
                model_full.fit(
                    X_full_train, y_train,
                    categorical_feature=cat_features_full
                )
                pred_proba_full = model_full.predict_proba(X_full_test)

                rms = log_loss(y_test, pred_proba_reduced, labels=np.arange(target_cardinality))
                fms = log_loss(y_test, pred_proba_full, labels=np.arange(target_cardinality))
                
                print(rms)
                print(fms)
                print()

                reduced_model_scores.append(rms)
                full_model_scores.append(fms)

            if len(full_model_scores) < 2:
                return False
            
            return should_keep_full_model(full_model_scores, reduced_model_scores, alpha)

        except (np.linalg.LinAlgError, ValueError, IndexError, PerfectSeparationError):
            return False"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import lightgbm as lgb
import warnings

def should_use_separate_models(mixed_scores, separate_scores, alpha):
    if np.allclose(np.array(mixed_scores), np.array(separate_scores[:len(mixed_scores)])) and np.allclose(np.array(mixed_scores), np.array(separate_scores[len(mixed_scores):])):
        print("→ IDENTICAL SCORES. Use mixed model.")
        return False
    
    #print(f"Mixed: {mixed_scores}")
    #print(f"Separate: {separate_scores}")
    t_stat, p_value = ttest_ind(mixed_scores, separate_scores, alternative='greater', equal_var=False)
    
    print(f"→ p-value: {p_value:.2e}")
    
    return p_value < alpha

def compute_cv_scores(data, Vk, B, categorical_features_B, n_splits, lgbm_params_base):
    y, class_names = pd.factorize(data[Vk])
    target_cardinality = len(class_names)
    if target_cardinality < 2:
        return None  # Signal to skip

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
                'n_jobs': -1,
                'class_weight': 'balanced'
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

            return should_use_separate_models(mixed_scores, separate_scores, alpha)

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