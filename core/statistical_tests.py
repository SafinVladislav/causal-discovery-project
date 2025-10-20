import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, chi2
from statsmodels.genmod.families import Binomial
import warnings

def marginal_test(Vk, data_obs, data_exp, alpha=0.05):
    obs_counts = data_obs[Vk].value_counts().sort_index()
    exp_counts = data_exp[Vk].value_counts().sort_index()

    all_categories = sorted(set(obs_counts.index) | set(exp_counts.index))
    obs_counts = obs_counts.reindex(all_categories, fill_value=0)
    exp_counts = exp_counts.reindex(all_categories, fill_value=0)

    if obs_counts.sum() < 50 or exp_counts.sum() < 50:
        return None

    contingency_table = np.array([obs_counts.values, exp_counts.values])

    expected_counts = chi2_contingency(contingency_table)[3]

    low_expected_count_percentage = np.mean(expected_counts < 5) * 100

    is_chi2_valid = (low_expected_count_percentage <= 20) and (np.all(expected_counts >= 1))

    if not is_chi2_valid:
        return False
    else:
        _, p_value, _, _ = chi2_contingency(contingency_table)
        return p_value < alpha

def conditional_test(Vk, B, data_obs, data_exp, alpha=0.05):
    if not B:
        return False

    if len(data_obs) < 50 or len(data_exp) < 50:
        return False

    combined_data = pd.concat([
        data_obs.assign(dataset_group=0),
        data_exp.assign(dataset_group=1)
    ])

    target_cardinality = combined_data[Vk].nunique()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            full_formula = f'{Vk} ~ dataset_group + {" + ".join(B)} + {" + ".join([f"{b} * dataset_group" for b in B])}'
            reduced_formula = f'{Vk} ~ {" + ".join(B)}'
            if target_cardinality <= 2:
                full_model = sm.GLM.from_formula(full_formula, data=combined_data, family=Binomial()).fit()
                reduced_model = sm.GLM.from_formula(reduced_formula, data=combined_data, family=Binomial()).fit()
            else:
                full_model = sm.MNLogit.from_formula(full_formula, data=combined_data).fit(disp=0)
                reduced_model = sm.MNLogit.from_formula(reduced_formula, data=combined_data).fit(disp=0)

            if not full_model.converged or not reduced_model.converged:
                return False

            lr_stat = 2 * (full_model.llf - reduced_model.llf)
            df = full_model.df_model - reduced_model.df_model
            p_value = chi2.sf(lr_stat, df=df)
            
            return p_value < alpha

    except (np.linalg.LinAlgError, ValueError, IndexError):
        return False

def robust_orientation_test(Vi, Vk, B, data_obs, data_exp, alpha1=0.05, alpha2=0.05):
    marginal_sig = marginal_test(Vk, data_obs, data_exp, alpha1)
    conditional_sig = conditional_test(Vk, B, data_obs, data_exp, alpha2)

    marg = marginal_sig > 0
    cond = conditional_sig > 0

    if marginal_sig and conditional_sig:
        return None, marg, cond
    elif marginal_sig:
        return (Vi, Vk), marg, cond
    elif conditional_sig:
        return (Vk, Vi), marg, cond
    else:
        return None, marg, cond