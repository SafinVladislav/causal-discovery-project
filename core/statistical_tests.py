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

    print("===")
    #print("Observational:")
    #print(data_obs[:5])
    #print("Experimental:")
    #print(data_exp[:5])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        try:
            # --- 2. Data Preparation ---
            combined_data = pd.concat([
                data_obs.assign(dataset_group=0),
                data_exp.assign(dataset_group=1)
            ]).copy() # Use .copy() to avoid SettingWithCopyWarning

            # Factorize the target variable (e.g., 'A', 'B', 'C' -> 0, 1, 2)
            y, class_names = pd.factorize(combined_data[Vk])
            target_cardinality = len(class_names)

            print(f"Cardinality: {target_cardinality}")

            if target_cardinality < 2:
                print("Constant target")
                # Target variable is constant, no test possible
                return False

            # Identify categorical features in B
            categorical_features_B = [
                col for col in B 
                if combined_data[col].dtype in ['object', 'category', 'bool']
            ]
            
            # Define feature sets for our two models
            features_reduced = B
            features_full = B + ['dataset_group']
            
            # LGBM works best with 'category' dtype
            cat_features_reduced = categorical_features_B
            cat_features_full = categorical_features_B + ['dataset_group']

            for col in cat_features_full:
                combined_data[col] = combined_data[col].astype('category')

            X_reduced = combined_data[features_reduced]
            X_full = combined_data[features_full]

            # --- 3. Cross-Validation Loop ---
            # Use StratifiedKFold to ensure class balance in each fold
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            reduced_model_scores = []
            full_model_scores = []

            # Define model parameters
            # These are lightweight for speed, as we are fitting 2*n_splits models
            lgbm_params = {
                'objective': 'binary' if target_cardinality == 2 else 'multiclass',
                'num_class': target_cardinality if target_cardinality > 2 else 1,
                'n_estimators': 50,
                'learning_rate': 0.1,
                'verbose': -1,
                'n_jobs': -1,
            }

            for train_idx, test_idx in skf.split(combined_data, y):
                # Get data for this fold
                X_reduced_train, X_reduced_test = X_reduced.iloc[train_idx], X_reduced.iloc[test_idx]
                X_full_train, X_full_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # --- 4. Fit Reduced Model (Vk ~ B) ---
                model_reduced = lgb.LGBMClassifier(**lgbm_params)
                model_reduced.fit(
                    X_reduced_train, y_train,
                    categorical_feature=cat_features_reduced
                )
                pred_proba_reduced = model_reduced.predict_proba(X_reduced_test)

                # --- 5. Fit Full Model (Vk ~ B + dataset_group) ---
                # The tree model will *implicitly* find interactions
                model_full = lgb.LGBMClassifier(**lgbm_params)
                model_full.fit(
                    X_full_train, y_train,
                    categorical_feature=cat_features_full
                )
                pred_proba_full = model_full.predict_proba(X_full_test)
                
                # --- 6. Score models ---
                # Check if all classes are present in the test set

                #print(f"{len(np.unique(y_test))}; {target_cardinality}")
                if len(np.unique(y_test)) == target_cardinality:
                    # Use log_loss: lower is better
                    print("Losses:")
                    print(log_loss(y_test, pred_proba_reduced, labels=class_names))
                    print(log_loss(y_test, pred_proba_full, labels=class_names))

                    reduced_model_scores.append(log_loss(y_test, pred_proba_reduced, labels=class_names))
                    full_model_scores.append(log_loss(y_test, pred_proba_full, labels=class_names))

            # --- 7. Statistical Test ---
            # We need at least 2 paired scores to run a t-test
            if len(full_model_scores) < 2:
                # print("Not enough valid folds to run t-test.")
                return False

            # We use a paired t-test because the scores are from the same CV folds
            # H0: full_loss >= reduced_loss (full model is not better)
            # Ha: full_loss < reduced_loss (full model IS better)
            t_stat, p_value = ttest_rel(
                full_model_scores, 
                reduced_model_scores, 
                alternative='less' # Test if full_scores are 'less than' reduced_scores
            )

            print(f"p_value: {p_value}")
            
            # print(f"Cond (Tree Test): {p_value}")
            return p_value < alpha

        except (np.linalg.LinAlgError, ValueError, IndexError, PerfectSeparationError):
            # Catch any fitting or data slicing errors
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