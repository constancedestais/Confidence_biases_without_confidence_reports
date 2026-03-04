# ANOVAs
import numpy as np 
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from Functions.text_p_value import text_p_value


def ANOVA_effect_of_click_desired_and_identify_best(merged_df):

    #%% sanity checks
    # chect that identify_best and click_desired columns exist, and that they have 2 unique values each
    if 'identify_best' not in merged_df.columns or 'click_desired' not in merged_df.columns:
        raise ValueError("Columns 'identify_best' and/or 'click_desired' not found in the dataframe.")
    if len(merged_df['identify_best'].unique()) < 2 or len(merged_df['click_desired'].unique()) < 2:
        raise ValueError("Columns 'identify_best' and/or 'click_desired' do not have enough unique values for ANOVA.")  
    # check that CFC_chose_highest_expected_value column exists
    if 'CFC_chose_highest_expected_value' not in merged_df.columns:
        raise ValueError("Column 'CFC_chose_highest_expected_value' not found in the dataframe.")
    

    #%% run two-way ANOVA
    model = ols('CFC_chose_highest_expected_value ~ C(identify_best) * C(click_desired)', data=merged_df).fit()
    anova_result = sm.stats.anova_lm(model, type=2)

    ''' 
    # double check with ttests
    # 1. identify_best: yes vs no
    group_identify_best_1 = merged_df[merged_df['identify_best'] == 1]['CFC_chose_highest_expected_value']
    group_identify_best_0 = merged_df[merged_df['identify_best'] == 0]['CFC_chose_highest_expected_value']
    tstat, pval_identify_best, df = sm.stats.ttest_ind(group_identify_best_1, group_identify_best_0)

    # 2. click_desired: yes vs no
    group_click_desired_1 = merged_df[merged_df['click_desired'] == 1]['CFC_chose_highest_expected_value']
    group_click_desired_0 = merged_df[merged_df['click_desired'] == 0]['CFC_chose_highest_expected_value']
    tstat, pval_click_desired, df = sm.stats.ttest_ind(group_click_desired_1, group_click_desired_0)
    
    print(f"T-test identify_best p-value: {pval_identify_best}")
    print(f"T-test click_desired p-value: {pval_click_desired}")
    '''

    # print general results
    print('\n----- ANOVA: CFC_chose_highest_expected_value ~ C(identify_best) * C(click_desired) ----- \n',anova_result)

    #%% print results fromatted for paper

    # print main effect of C(identify_best) ("CFC manipulation") with effect size F, df and pvalue
    F_identify_best = anova_result.loc['C(identify_best)', 'F']
    df1_identify_best = int(anova_result.loc['C(identify_best)', 'df'])
    df2_identify_best = int(anova_result.loc['Residual', 'df'])
    pvalue_identify_best = anova_result.loc['C(identify_best)', 'PR(>F)']
    pvalue_identify_best_txt = text_p_value(pvalue_identify_best)
    print(f"\nANOVA CFC framing effect on p_chose_highest_EV: F({df1_identify_best},{df2_identify_best})={F_identify_best:.2f}, {pvalue_identify_best_txt};")

    # print main effect of C(click_desired) ("LT manipulation") with effect size F, df and pvalue
    F_click_desired = anova_result.loc['C(click_desired)', 'F']
    df1_click_desired = int(anova_result.loc['C(click_desired)', 'df'])
    df2_click_desired = int(anova_result.loc['Residual', 'df'])         
    pvalue_click_desired = anova_result.loc['C(click_desired)', 'PR(>F)']
    pvalue_click_desired_txt = text_p_value(pvalue_click_desired)
    print(f"\nANOVA LT framing effect on p_chose_highest_EV: F({df1_click_desired},{df2_click_desired})={F_click_desired:.2f}, {pvalue_click_desired_txt};")

    # print interaction effect with effect size F, df and pvalue
    F_interaction = anova_result.loc['C(identify_best):C(click_desired)', 'F']
    df1_interaction = int(anova_result.loc['C(identify_best):C(click_desired)', 'df'])
    df2_interaction = int(anova_result.loc['Residual', 'df'])         
    pvalue_interaction = anova_result.loc['C(identify_best):C(click_desired)', 'PR(>F)']
    pvalue_interaction_txt = text_p_value(pvalue_interaction)
    print(f"\nANOVA interaction effect on p_chose_highest_EV: F({df1_interaction},{df2_interaction})={F_interaction:.2f}, {pvalue_interaction_txt};\n")  

    return




def ANOVA_effect_of_identify_best_and_asymmetric_difficulty(merged_df):
    #%% checks

    # check that difficulty column exists
    if 'LT_unequal_difficulty_binary' not in merged_df.columns:
        raise ValueError("Column 'LT_unequal_difficulty_binary' not found in the dataframe.")
    # chect that identify_best and click_desired columns exist, and that they have 2 unique values each
    if 'identify_best' not in merged_df.columns :
        raise ValueError("Columns 'identify_best' not found in the dataframe.")
    if len(merged_df['identify_best'].unique()) < 2:
        raise ValueError("Columns 'identify_best' do not have enough unique values for ANOVA.")  
    # check that CFC_chose_highest_expected_value column exists
    if 'CFC_chose_highest_expected_value' not in merged_df.columns:
        raise ValueError("Column 'CFC_chose_highest_expected_value' not found in the dataframe.")
    
    # check that data includes versions with and without difficulty manipulation
    equal_difficulty_versions = ['cd1_2025_click_desired_1_identify_best_1', 'cd1_2025_click_desired_1_identify_best_0', 'cd1_2025_click_desired_0_identify_best_1', 'cd1_2025_click_desired_0_identify_best_0']
    unequal_difficulty_versions = ['cd1_2025_click_desired_1_identify_best_0_difficulty_0_70_0_80', 'cd1_2025_click_desired_1_identify_best_1_difficulty_0_70_0_80']
    if not any(merged_df['exp_ID'].isin(unequal_difficulty_versions)):  
        raise ValueError("Data does not include versions with difficulty manipulation required for this ANOVA.")
    if not any(merged_df['exp_ID'].isin(equal_difficulty_versions)):  
        raise ValueError("Data does not include versions without difficulty manipulation required for this ANOVA.")
    
    #%% run three-way ANOVA
    model = ols('CFC_chose_highest_expected_value ~ C(identify_best) * C(LT_unequal_difficulty_binary)', data=merged_df).fit()
    anova_result = sm.stats.anova_lm(model, type=2) 
    
    # print general results
    print('\n----- ANOVA: CFC_chose_highest_expected_value ~ C(identify_best) * C(LT_unequal_difficulty_binary) ----- \n',anova_result)

    #%% print results fromatted for paper

    # print main effect of C(identify_best) ("CFC manipulation") with effect size F, df and pvalue
    F_identify_best = anova_result.loc['C(identify_best)', 'F']
    df1_identify_best = int(anova_result.loc['C(identify_best)', 'df'])
    df2_identify_best = int(anova_result.loc['Residual', 'df'])
    pvalue_identify_best = anova_result.loc['C(identify_best)', 'PR(>F)']
    pvalue_identify_best_txt = text_p_value(pvalue_identify_best)
    print(f"\nANOVA CFC framing effect on p_chose_highest_EV: F({df1_identify_best},{df2_identify_best})={F_identify_best:.2f}, {pvalue_identify_best_txt};")

    # print main effect of C(LT_unequal_difficulty_binary) ("LT manipulation") with effect size F, df and pvalue
    F_difficulty = anova_result.loc['C(LT_unequal_difficulty_binary)', 'F']
    df1_difficulty = int(anova_result.loc['C(LT_unequal_difficulty_binary)', 'df'])
    df2_difficulty = int(anova_result.loc['Residual', 'df'])         
    pvalue_difficulty = anova_result.loc['C(LT_unequal_difficulty_binary)', 'PR(>F)']
    pvalue_difficulty_txt = text_p_value(pvalue_difficulty)
    print(f"\nANOVA asymmetric difficulty effect on p_chose_highest_EV: F({df1_difficulty},{df2_difficulty})={F_difficulty:.2f}, {pvalue_difficulty_txt};")

    # print interaction effect with effect size F, df and pvalue
    F_interaction = anova_result.loc['C(identify_best):C(LT_unequal_difficulty_binary)', 'F']
    df1_interaction = int(anova_result.loc['C(identify_best):C(LT_unequal_difficulty_binary)', 'df'])
    df2_interaction = int(anova_result.loc['Residual', 'df'])         
    pvalue_interaction = anova_result.loc['C(identify_best):C(LT_unequal_difficulty_binary)', 'PR(>F)']
    pvalue_interaction_txt = text_p_value(pvalue_interaction)
    print(f"\nANOVA interaction effect on p_chose_highest_EV: F({df1_interaction},{df2_interaction})={F_interaction:.2f}, {pvalue_interaction_txt};\n")  

    return



def ANOVA_effect_of_pair_composition_on_pChose_highest_EV(CFC_mean_per_participant_by_pair_valence_composition_and_includes_new_pair):
    '''
    # ANOVA not possible because design is unbalanced (cannot have heterogeneous valence pairs without new pairs)

    # check if data is unbalanced before repeated measures ANOVA
    counts = CFC_mean_per_participant_by_pair_valence_composition_and_includes_new_pair.groupby(['participant_ID']).size()
    if counts.nunique() > 1:
        raise ValueError("Data is unbalanced: not all participants have data for all conditions required for repeated measures ANOVA.")
    # repeated measures ANOVA
    anova_result = sm.stats.AnovaRM(data=CFC_mean_per_participant_by_pair_valence_composition_and_includes_new_pair, 
                                    depvar="chose_highest_expected_value", 
                                    subject="participant_ID",
                                    within=["includes_new_pair","pair_valence_composition"])
    print('\n----- ANOVA w/ repeated measures: chose_highest_expected_value ~ C(includes_new_pair) * C(pair_valence_composition) -----')
    print(anova_result.fit())

    '''

    return    
