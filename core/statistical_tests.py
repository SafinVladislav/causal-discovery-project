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
            combined_data['random_binary'] = np.random.randint(0, 2, size=combined_data.shape[0])

            y, class_names = pd.factorize(combined_data[Vk])

            target_cardinality = len(class_names)
            if target_cardinality < 2:
                return False

            categorical_features_B = [
                col for col in B 
                if combined_data[col].dtype in ['object', 'category', 'bool']
            ]
            
            features_reduced = B + ['random_binary']
            features_full = B + ['dataset_group']
            
            cat_features_reduced = categorical_features_B + ['random_binary']
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
                'n_jobs': -1
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

                reduced_model_scores.append(log_loss(y_test, pred_proba_reduced, labels=np.arange(target_cardinality)))
                full_model_scores.append(log_loss(y_test, pred_proba_full, labels=np.arange(target_cardinality)))

            if len(full_model_scores) < 2:
                return False

            if np.allclose(np.array(full_model_scores) - np.array(reduced_model_scores), 0):
                p_value = 1.0 
            else:
                t_stat, p_value = ttest_rel(
                    full_model_scores, 
                    reduced_model_scores, 
                    alternative='less'
                )
            return p_value < alpha

        except (np.linalg.LinAlgError, ValueError, IndexError, PerfectSeparationError):
            return False

N_SPLITS = 5

def robust_orientation_test(Vi, Vk, B, data_obs, data_exp, alpha1, alpha2):
    marg = marginal_test(Vk, data_obs, data_exp, alpha1)
    if marg:
        return (Vi, Vk)

    cond = conditional_test(Vk, B, data_obs, data_exp, alpha2, N_SPLITS)
    if cond:
        return (Vk, Vi)

    return None