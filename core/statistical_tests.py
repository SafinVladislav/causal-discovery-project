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

def marginal_test(Vk, data_obs, data_exp, alpha=0.05):
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

    if not is_chi2_valid:
        return False
    else:
        #print(f"Marg: {p_value}")

        return p_value < alpha

def conditional_test(Vk, B, data_obs, data_exp, alpha=0.05, n_splits=5):
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
            #print("===")
            print("Parent Class Occurrences (obs):")
            print(data_obs[B[0]].value_counts())
            print("Parent Class Occurrences (exp):")
            print(data_exp[B[0]].value_counts())
            print("Class Occurrences (obs):")
            print(data_obs[Vk].value_counts())
            print("Class Occurrences (exp):")
            print(data_exp[Vk].value_counts())

            print(target_cardinality)

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
                
                print(log_loss(y_test, pred_proba_reduced, labels=np.arange(target_cardinality)))
                print(log_loss(y_test, pred_proba_full, labels=np.arange(target_cardinality)))
                print()
                reduced_model_scores.append(log_loss(y_test, pred_proba_reduced, labels=np.arange(target_cardinality)))
                full_model_scores.append(log_loss(y_test, pred_proba_full, labels=np.arange(target_cardinality)))

            if len(full_model_scores) < 2:
                return False

            if np.allclose(np.array(full_model_scores) - np.array(reduced_model_scores), 0):
                # If all differences are zero, the models performed identically.
                # The t-statistic would be 0, and the p-value is 1.0.
                p_value = 1.0 
            else:
                # Only run the t-test if there is variance in the difference scores
                t_stat, p_value = ttest_rel(
                    full_model_scores, 
                    reduced_model_scores, 
                    alternative='less'
                )

            print(p_value)
            
            return p_value < alpha

        except (np.linalg.LinAlgError, ValueError, IndexError, PerfectSeparationError):
            return False

def robust_orientation_test(Vi, Vk, B, data_obs, data_exp, alpha1=0.05, alpha2=0.05):
    marg = False#marginal_test(Vk, data_obs, data_exp, alpha1)
    #marginal_test(Vk, data_obs, data_exp, alpha1)
    cond = conditional_test(Vk, B, data_obs, data_exp, alpha2)

    #print(cond)

    if marg and cond:
        return None, marg, cond
    elif marg:
        return (Vi, Vk), marg, cond
    elif cond:
        return (Vk, Vi), marg, cond
    else:
        return None, marg, cond