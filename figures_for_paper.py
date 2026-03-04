#%% script to make figures that contain data from multiple datasets

#%% import packages & set paths & set directories

# import packages
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from Functions.plot_chose_gain_pair_CFC import plot_chose_gain_pair_CFC
from Functions.map_to_symbols_1_and_2 import *
from Functions.create_expected_value_column import *
from Functions.create_is_gain_trial_column import *
from Functions.get_version_code import get_version_code
from Functions.filter_experiment_version import filter_experiment_version
from Functions.plot_regressions_CFC_vs_LT_SC import *
from Functions.plot_regressions_model_comparison  import (plot_regressions_model_comparison_RLtreatment_CFCtreatment, 
                                                          plot_regressions_model_comparison_difficulty_CFCtreatment,
                                                          plot_regressions_model_comparison_CFCtreatment )
from Functions.ANOVAs_chose_gain_pair import (ANOVA_effect_of_click_desired_and_identify_best, 
                                                ANOVA_effect_of_identify_best_and_asymmetric_difficulty, 
                                                )
from Functions.prepare_data_for_figures_comparing_multiple_datasets import (prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets, 
                                                                            prepare_CFC_data_by_pair_composition_for_figures_comparing_multiple_datasets)
from Functions.plot_chose_gain_pair_CFC_for_heterogeneous_vs_homogeneous_valence_pairs import plot_chose_gain_pair_CFC_for_heterogeneous_vs_homogeneous_valence_pairs
from Functions.plot_chose_gain_pair_CFC_for_new_vs_original_pairs import plot_chose_gain_pair_CFC_for_new_vs_original_pairs
from Functions.plot_pcorrect_valence_over_trials_LT import plot_pcorrect_valence_over_trials_LT
from Functions.plot_pcorrect_valence_LT import plot_pcorrect_valence_LT
from Functions.plot_pcorrect_valence_SC import plot_pcorrect_valence_SC
from Functions.scale_0_1_to_0_100 import scale_0_1_to_0_100

# set current path
path = os.getcwd()
print(path)
output_dir = os.path.join(path, "Outputs")
figures_dir = os.path.join(output_dir, "Figures")
data_dir = os.path.join(path, "Data")
raw_data_dir = os.path.join(data_dir, 'raw_data')

#%% load all data
# specify the name of the data folder - depends on date of data collection
current_data_dir = os.path.join(data_dir)

# specify the filenames of the CSV files
csv_filenames = {
    'LearningTask': 'CD1_LearningTask.csv',
    'PairChoice'  : 'CD1_PairChoice.csv',
    'SymbolChoice': 'CD1_SymbolChoice.csv',
    'BonusRound'  : 'CD1_BonusRound.csv',
    'Demographics': 'CD1_Demographics.csv',
    'General'     : 'CD1_General.csv'
}

# create a dictionary to store the dataframes
dataframes = {}
# read the CSV files into the dictionary
for key, filename in csv_filenames.items():
    csv_file_path = os.path.join(current_data_dir, filename)
    dataframes[key] = pd.read_csv(csv_file_path)


#prep matplotlib for SVG output with text as text (not paths) for compatibility with Inkscape

mpl.rcParams["svg.fonttype"] = "none"        # <-- keep text as text
mpl.rcParams["text.usetex"] = False          # TeX often forces path-like output


# ======================================================================================================================
# ======================================================================================================================

#%% model comparison including with variables CFCtreatment and RLtreatment in versions_equal_difficulty_across_gain_loss 

requested_subset = 'versions_equal_difficulty_across_gain_loss'
# prep data - filter to only include datasets with equal difficulty across gain and loss
dataframes_reduced = filter_experiment_version(dataframes.copy(), requested_subset)
version_code = get_version_code(requested_subset)
[merged_data] = prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets(dataframes_reduced['LearningTask'], 
                                                                                        dataframes_reduced['SymbolChoice'], 
                                                                                        dataframes_reduced['PairChoice'], 
                                                                                        dataframes_reduced['Demographics'])

# check that dependent and independent variables are not between 0 and 1
assert merged_data["CFC_chose_highest_expected_value"].max() > 1.5, "Error: CFC_chose_highest_expected_value values look like they are still coded 0-1"
assert merged_data['LT_correct_gain_minus_loss'].max() > 1.5, f"Error: {'LT_correct_gain_minus_loss'} values look like they are still coded 0-1" 
assert merged_data['SC_correct_gain_minus_loss'].max() > 1.5, f"Error: {'SC_correct_gain_minus_loss'} values look like they are still coded 0-1" 

# plot
plot_regressions_model_comparison_RLtreatment_CFCtreatment(
    merged_data=merged_data,
    version_code=version_code,
    independent_variable='LT_correct_gain_minus_loss')

plot_regressions_model_comparison_RLtreatment_CFCtreatment(
    merged_data=merged_data,
    version_code=version_code,
    independent_variable='SC_correct_gain_minus_loss')



#%% model comparison for difficulty treatment and CFC treatment in versions_equal_and_asymmetric_difficulty_click_desired_1
requested_subset = 'versions_equal_and_asymmetric_difficulty_click_desired_1'
# prep data - filter to only include datasets with equal difficulty across gain and loss
dataframes_reduced = filter_experiment_version(dataframes.copy(), requested_subset)
version_code = get_version_code(requested_subset)
[merged_data] = prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets(dataframes_reduced['LearningTask'], 
                                                                                        dataframes_reduced['SymbolChoice'], 
                                                                                        dataframes_reduced['PairChoice'], 
                                                                                        dataframes_reduced['Demographics'])

# check that dependent and independent variables are not between 0 and 1
assert merged_data["CFC_chose_highest_expected_value"].max() > 1.5, "Error: CFC_chose_highest_expected_value values look like they are still coded 0-1"
assert merged_data['LT_correct_gain_minus_loss'].max() > 1.5, f"Error: {'LT_correct_gain_minus_loss'} values look like they are still coded 0-1" 
assert merged_data['SC_correct_gain_minus_loss'].max() > 1.5, f"Error: {'SC_correct_gain_minus_loss'} values look like they are still coded 0-1" 

# model comparison
plot_regressions_model_comparison_difficulty_CFCtreatment(
    merged_data=merged_data,
    version_code=version_code,
    independent_variable='LT_correct_gain_minus_loss'
)
plot_regressions_model_comparison_difficulty_CFCtreatment(
    merged_data=merged_data,
    version_code=version_code,
    independent_variable='SC_correct_gain_minus_loss'
)


#%% model comparison for CFC treatment in versions_asymmetric_difficulty_across_gain_loss

requested_subset = 'versions_asymmetric_difficulty_across_gain_loss'
# prep data - filter to only include datasets with equal difficulty across gain and loss
dataframes_reduced = filter_experiment_version(dataframes.copy(), requested_subset)
version_code = get_version_code(requested_subset)
[merged_data] = prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets(dataframes_reduced['LearningTask'], 
                                                                                        dataframes_reduced['SymbolChoice'], 
                                                                                        dataframes_reduced['PairChoice'], 
                                                                                        dataframes_reduced['Demographics'])

# check that dependent and independent variables are not between 0 and 1
assert merged_data["CFC_chose_highest_expected_value"].max() > 1.5, "Error: CFC_chose_highest_expected_value values look like they are still coded 0-1"
assert merged_data['LT_correct_gain_minus_loss'].max() > 1.5, f"Error: {'LT_correct_gain_minus_loss'} values look like they are still coded 0-1" 
assert merged_data['SC_correct_gain_minus_loss'].max() > 1.5, f"Error: {'SC_correct_gain_minus_loss'} values look like they are still coded 0-1" 

# model comparison
plot_regressions_model_comparison_CFCtreatment(
    merged_data=merged_data,
    version_code=version_code,
    independent_variable='LT_correct_gain_minus_loss'
)
plot_regressions_model_comparison_CFCtreatment(
    merged_data=merged_data,
    version_code=version_code,
    independent_variable='SC_correct_gain_minus_loss'
)







#%% plot accuracy[gain-loss] across each version

exp_versions = ['cd1_2025_click_desired_1_identify_best_1', 
                'cd1_2025_click_desired_1_identify_best_0',
                'cd1_2025_click_desired_0_identify_best_1',
                'cd1_2025_click_desired_0_identify_best_0',
                'cd1_2025_click_desired_1_identify_best_1_difficulty_0_70_0_80',
                'cd1_2025_click_desired_1_identify_best_0_difficulty_0_70_0_80']

# loop over each experiment version
for exp_version in exp_versions:
    dataframes_copy = dataframes.copy()

    # filter by exp version
    dataframes_reduced = filter_experiment_version(dataframes_copy, exp_version) 
    # use assert to check only one exp version is present
    assert len(dataframes_reduced['LearningTask']['exp_ID'].unique()) == 1, "More than one experiment version found after filtering."
    # get shorter code for experiment version for filenames   
    version_code = get_version_code( dataframes_reduced['LearningTask']['exp_ID'].unique() )

    # DEBUGGING - print unique expID
    print(f"\nExperiment version: {dataframes_reduced['LearningTask']['exp_ID'].unique()[0]}\n")

    # ------- plot pcorrect over trials in LT, separately for gain and loss trials -------
    LT_mean_per_participant_per_trial_per_valence = dataframes_reduced['LearningTask'].groupby(['participant_ID', 'trial_per_pair', 'is_gain_trial'], as_index=False)['correct'].mean()
    # scale to percentage
    LT_mean_per_participant_per_trial_per_valence["correct"] = scale_0_1_to_0_100(LT_mean_per_participant_per_trial_per_valence["correct"], label="LT correct")
    # plot
    fig, ax = plot_pcorrect_valence_over_trials_LT(LT_mean_per_participant_per_trial_per_valence)
    fig.savefig('./Outputs/Figures/pcorrect_valence_across_trials_LT_'+version_code+'.svg',bbox_inches='tight')
    del fig, ax

    # ------- plot pcorrect in LT, separately for gain and loss -------
    LT_mean_per_participant_per_valence = dataframes_reduced['LearningTask'].groupby(['participant_ID', 'is_gain_trial'], as_index=False)['correct'].mean()
    # scale to percentage
    LT_mean_per_participant_per_valence["correct"] = scale_0_1_to_0_100(LT_mean_per_participant_per_valence["correct"], label="LT correct")
    # plot
    fig, ax = plot_pcorrect_valence_LT(LT_mean_per_participant_per_valence)
    fig.savefig('./Outputs/Figures/pcorrect_valence_LT_'+version_code+'.svg',bbox_inches='tight')
    del fig, ax
    # DEBUGGING ttest 1
    gain_trials = LT_mean_per_participant_per_valence[LT_mean_per_participant_per_valence['is_gain_trial'] == 1]['correct']
    loss_trials = LT_mean_per_participant_per_valence[LT_mean_per_participant_per_valence['is_gain_trial'] == 0]['correct']
    t_stat, p_value = stats.ttest_rel(gain_trials, loss_trials)
    df = len(gain_trials) - 1
    print(f"\nT-test for LT accuracy in gain vs. loss trials: t({df})={t_stat}, p={p_value}") 
    # DEBUGGING ttest 2
    wide = LT_mean_per_participant_per_valence.pivot(index="participant_ID", columns="is_gain_trial", values="correct")
    paired = wide.dropna(subset=[False, True])
    t_stat, p_value = stats.ttest_rel(paired[True], paired[False])
    df = len(paired) - 1
    print(f"T-test ... (wide): t({df})={t_stat}, p={p_value}")

    # ------- plot pcorrect in SC, separately for gain and loss -------
    SC_mean_per_participant_per_valence = dataframes_reduced['SymbolChoice'].groupby(['participant_ID', 'is_gain_trial'], as_index=False)['correct'].mean()
    # scale to percentage
    SC_mean_per_participant_per_valence["correct"] = scale_0_1_to_0_100(SC_mean_per_participant_per_valence["correct"], label="SC correct")
    # plot
    fig, ax = plot_pcorrect_valence_SC(SC_mean_per_participant_per_valence)
    fig.savefig('./Outputs/Figures/pcorrect_valence_SC_'+version_code+'.svg',bbox_inches='tight')
    del fig, ax
    # DEBUGGING ttest
    gain_trials = SC_mean_per_participant_per_valence[SC_mean_per_participant_per_valence['is_gain_trial'] == 1]['correct']
    loss_trials = SC_mean_per_participant_per_valence[SC_mean_per_participant_per_valence['is_gain_trial'] == 0]['correct']
    t_stat, p_value = stats.ttest_rel(gain_trials, loss_trials)
    df = len(gain_trials) - 1
    print(f"\nT-test for SC accuracy in gain vs. loss trials: t({df})={t_stat}, p={p_value}") 

    # ------- plot chose gain pair in CFC -------
    CFC_mean_per_participant = dataframes_reduced['PairChoice'].groupby(['participant_ID'], as_index=False)['chose_highest_expected_value'].mean()
    # scale to percentage
    CFC_mean_per_participant["chose_highest_expected_value"] = scale_0_1_to_0_100(CFC_mean_per_participant["chose_highest_expected_value"], label="CFC chose highest expected value")
    # plot
    fig, ax = plot_chose_gain_pair_CFC(CFC_mean_per_participant)
    fig.savefig('./Outputs/Figures/chose_gain_pair_CFC_'+version_code+'.svg',bbox_inches='tight')
    plt.close(fig)
    del fig, ax
    # DEBUGGING ttest
    t_stat, p_value = stats.ttest_1samp(CFC_mean_per_participant['chose_highest_expected_value'], 50)
    print(f"One-sample t-test against 50 for chose_highest_expected_value in CFC: t({len(CFC_mean_per_participant)-1})={t_stat:.2f}, p={p_value:.3f};")
    a=1
  





#%% plot effect of pair composition (within participant) on choice in CFC, averaged first over versions_equal_difficulty_across_gain_loss then over versions_asymmetric_difficulty_across_gain_loss
requested_subsets = ['versions_equal_difficulty_across_gain_loss','versions_asymmetric_difficulty_across_gain_loss']

for requested_subset in requested_subsets:
    dataframes_reduced = filter_experiment_version(dataframes.copy(), requested_subset)
    version_code = get_version_code(requested_subset)

    [unused,
    CFC_mean_per_participant_by_pair_valence_composition,
    CFC_mean_per_participant_by_includes_new_pair,
    unused] = prepare_CFC_data_by_pair_composition_for_figures_comparing_multiple_datasets(dataframes_reduced['PairChoice'])

    # look at effect of includes_new_pair (new vs. original pairs) on pChose Highest EV
    # check that variable is not between 0 and 1 (and thus is probably scaled to percentage)
    assert CFC_mean_per_participant_by_includes_new_pair['chose_highest_expected_value'].max() > 1.5, f"Error: chose_highest_expected_value values look like they are still coded 0-1" 
    new_pair_present = CFC_mean_per_participant_by_includes_new_pair[CFC_mean_per_participant_by_includes_new_pair['includes_new_pair'] == 1]['chose_highest_expected_value']
    new_pair_absent = CFC_mean_per_participant_by_includes_new_pair[CFC_mean_per_participant_by_includes_new_pair['includes_new_pair'] == 0]['chose_highest_expected_value']
    fig, ax = plot_chose_gain_pair_CFC_for_new_vs_original_pairs([new_pair_present.values, new_pair_absent.values])
    fig.set_size_inches(1.5, 1.5)
    fig.savefig('./Outputs/Figures/plot_CFC_chose_highest_EV_for_new_vs_original_pairs_'+version_code+'.svg')
    del fig, ax


    # look at effect of pair_valence_composition (heterogeneous vs. homogeneous) on pChose Highest EV
    # check that variable is not between 0 and 1 (and thus is probably scaled to percentage)
    assert CFC_mean_per_participant_by_pair_valence_composition['chose_highest_expected_value'].max() > 1.5, f"Error: chose_highest_expected_value values look like they are still coded 0-1" 
    heterogeneous_valence = CFC_mean_per_participant_by_pair_valence_composition[CFC_mean_per_participant_by_pair_valence_composition['pair_valence_composition'] == "heterogeneous_symbol_valence"]['chose_highest_expected_value']
    homogeneous_valence = CFC_mean_per_participant_by_pair_valence_composition[CFC_mean_per_participant_by_pair_valence_composition['pair_valence_composition'] == "homogeneous_symbol_valence"]['chose_highest_expected_value']
    fig, ax = plot_chose_gain_pair_CFC_for_heterogeneous_vs_homogeneous_valence_pairs([heterogeneous_valence.values, homogeneous_valence.values])
    fig.set_size_inches(1.5, 1.5)
    fig.savefig('./Outputs/Figures/plot_CFC_chose_highest_EV_for_heterogeneous_vs_homogeneous_valence_pairs_'+version_code+'.svg')
    del fig, ax



#%% regression plot restricted to CFC trials with only new pairs, first averaged over versions_equal_difficulty_across_gain_loss then over versions_asymmetric_difficulty_across_gain_loss

requested_subsets = ['versions_equal_difficulty_across_gain_loss','versions_asymmetric_difficulty_across_gain_loss']

for requested_subset in requested_subsets:
        
    dataframes_reduced = filter_experiment_version(dataframes.copy(), requested_subset)
    version_code = get_version_code(requested_subset)

    # filter PairChoice data, only keep rows for whcih n_new_pairs > 0  (i.e. trials with at least one new pair)
    PairChoice_new_pairs_only = dataframes_reduced['PairChoice'][dataframes_reduced['PairChoice']['n_new_pairs'] > 0].copy()
    # check that none of the pairs have n_new_pair == 0
    assert PairChoice_new_pairs_only['n_new_pairs'].min() == 1, "Error: some pairs in PairChoice_new_pairs_only have includes_new_pair == 0"
    # merge with learning task to get merged_data_new_pairs_only
    [merged_data_new_pairs_only] = prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets(dataframes_reduced['LearningTask'], 
                                                                                                            dataframes_reduced['SymbolChoice'], 
                                                                                                            PairChoice_new_pairs_only, 
                                                                                                            dataframes_reduced['Demographics'])

    # run regression plots
    independent_variables = ['LT_correct_gain_minus_loss', 'SC_correct_gain_minus_loss']
    task_names = ['LT','SC']  
    for independent_variable, task in zip(independent_variables, task_names):
        # check that dependent and independent variables is not between 0 and 1
        assert merged_data_new_pairs_only["CFC_chose_highest_expected_value"].max() > 1.5, "Error: CFC_chose_highest_expected_value values look like they are still coded 0-1"
        assert merged_data_new_pairs_only['LT_correct_gain_minus_loss'].max() > 1.5, f"Error: {'LT_correct_gain_minus_loss'} values look like they are still coded 0-1" 
        assert merged_data_new_pairs_only['SC_correct_gain_minus_loss'].max() > 1.5, f"Error: {'SC_correct_gain_minus_loss'} values look like they are still coded 0-1" 

        plot_winning_regression_line_over_all_data(
            merged_data=merged_data_new_pairs_only,
            independent_variable=independent_variable,
            filename=f'./Outputs/Figures/plot_winning_regression_CFC_vs_'+task+'_new_pairs_only_'+version_code+'.svg'
            )

        plot_winning_regression_parameters_all_data(
            merged_data = merged_data_new_pairs_only,
            independent_variable = independent_variable,
            filenames = ['./Outputs/Figures/winning_regression_formula_CFC_vs_'+task+'_new_pairs_only_'+version_code+'.txt',
                        './Outputs/Figures/plot_winning_regression_intercept_CFC_vs_'+task+'_new_pairs_only_'+version_code+'.svg',
                        './Outputs/Figures/plot_winning_regression_slope_CFC_vs_'+task+'_new_pairs_only_'+version_code+'.svg',
                        './Outputs/Figures/plot_winning_regression_coeff_CFC_vs_'+task+'_new_pairs_only_'+version_code+'.svg',
                        './Outputs/Figures/plot_winning_regression_two_intercepts_CFC_vs_'+task+'_new_pairs_only_'+version_code+'.svg',
                        './Outputs/Figures/plot_winning_regression_piechart_CFC_vs_'+task+'_new_pairs_only_'+version_code+'.svg'],
                )

    del merged_data_new_pairs_only







#%% regression plots for winning model : chose positive pair ~ accuracy[gain-loss] + CFC+/-; first for versions_equal_difficulty_across_gain_loss then versions_asymmetric_difficulty_across_gain_loss

requested_subsets = ['versions_equal_difficulty_across_gain_loss', 
                     'versions_asymmetric_difficulty_across_gain_loss']
for requested_subset in requested_subsets:  
    # prep data - filter to only include datasets with equal difficulty across gain and loss
    dataframes_reduced = filter_experiment_version(dataframes.copy(), requested_subset)
    version_code = get_version_code(requested_subset)
    [merged_data] = prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets(dataframes_reduced['LearningTask'], 
                                                                                            dataframes_reduced['SymbolChoice'], 
                                                                                            dataframes_reduced['PairChoice'], 
                                                                                            dataframes_reduced['Demographics'])


    # run regression plots
    independent_variables = ['LT_correct_gain_minus_loss', 'SC_correct_gain_minus_loss']
    task_names = ['LT','SC']  
    for independent_variable, task in zip(independent_variables, task_names):
        # check that dependent and independent variables are not between 0 and 1
        assert merged_data["CFC_chose_highest_expected_value"].max() > 1.5, "Error: CFC_chose_highest_expected_value values look like they are still coded 0-1"
        assert merged_data[independent_variable].max() > 1.5, f"Error: {independent_variable} values look like they are still coded 0-1" 
        # plot regressions
        plot_winning_regression_line_over_all_data(
            merged_data=merged_data,
            independent_variable=independent_variable,
            filename=f'./Outputs/Figures/plot_winning_regression_CFC_vs_'+task+'_'+version_code+'.svg'
            )
        plot_winning_regression_parameters_all_data(
            merged_data = merged_data,
            independent_variable = independent_variable,
            filenames = ['./Outputs/Figures/winning_regression_formula_CFC_vs_'+task+'_'+version_code+'.txt',
                        './Outputs/Figures/plot_winning_regression_intercept_CFC_vs_'+task+'_'+version_code+'.svg',
                        './Outputs/Figures/plot_winning_regression_slope_CFC_vs_'+task+'_'+version_code+'.svg',
                        './Outputs/Figures/plot_winning_regression_coeff_CFC_vs_'+task+'_'+version_code+'.svg',
                        './Outputs/Figures/plot_winning_regression_two_intercepts_CFC_vs_'+task+'_'+version_code+'.svg',
                        './Outputs/Figures/plot_winning_regression_piechart_CFC_vs_'+task+'_'+version_code+'.svg'],
            )
        a=1

del requested_subsets, dataframes_reduced, version_code
del merged_data



  
#%%  ANOVA results: effect of RL treatment x CFC treatment on chose gain pair in CFC in versions_equal_difficulty_across_gain_loss

# prep data - filter to only include datasets with equal difficulty across gain and loss
requested_subset = 'versions_equal_difficulty_across_gain_loss'
dataframes_reduced = filter_experiment_version(dataframes.copy(), requested_subset)
version_code = get_version_code(requested_subset)
[merged_data] = prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets(dataframes_reduced['LearningTask'], 
                                                                                        dataframes_reduced['SymbolChoice'], 
                                                                                        dataframes_reduced['PairChoice'], 
                                                                                        dataframes_reduced['Demographics'])
# check that variables are not between 0 and 1
assert merged_data["CFC_chose_highest_expected_value"].max() > 1.5, "Error: CFC_chose_highest_expected_value values look like they are still coded 0-1"
assert merged_data['LT_correct_gain_minus_loss'].max() > 1.5, f"Error: {'LT_correct_gain_minus_loss'} values look like they are still coded 0-1" 
assert merged_data['SC_correct_gain_minus_loss'].max() > 1.5, f"Error: {'SC_correct_gain_minus_loss'} values look like they are still coded 0-1"    

# run ANOVA
ANOVA_effect_of_click_desired_and_identify_best(merged_data)


#%% regression plots for 'cd1_2025_click_desired_1_identify_best_1'

# load all data, will then filter to only include one dataset (NOT EFFICIENT)
requested_subset = 'all'  
dataframes_reduced = filter_experiment_version(dataframes.copy(), requested_subset)
version_code = get_version_code(requested_subset)
[merged_data] = prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets(dataframes_reduced['LearningTask'], 
                                                                                         dataframes_reduced['SymbolChoice'], 
                                                                                         dataframes_reduced['PairChoice'], 
                                                                                         dataframes_reduced['Demographics'])


if 'cd1_2025_click_desired_1_identify_best_1' in merged_data['exp_ID'].unique():
    # filter merged data - one dataset only
    dataset = 'cd1_2025_click_desired_1_identify_best_1'
    merged_data_single_version = merged_data[merged_data['exp_ID'] == dataset].copy() 
    single_version_code = get_version_code( merged_data_single_version['exp_ID'].unique() )

    independent_variables = ['LT_correct_gain_minus_loss', 'SC_correct_gain_minus_loss']
    task_names = ['LT','SC']  
    for independent_variable, task in zip(independent_variables, task_names):
        # check that dependent and independent variables are not between 0 and 1
        assert merged_data["CFC_chose_highest_expected_value"].max() > 1.5, "Error: CFC_chose_highest_expected_value values look like they are still coded 0-1"
        assert merged_data[independent_variable].max() > 1.5, f"Error: {independent_variable} values look like they are still coded 0-1" 
        # plot regressions
        plot_simple_regression_line_for_one_dataset(
            merged_data=merged_data_single_version,
            independent_variable=independent_variable,
            filename=f'./Outputs/Figures/plot_regression_line_for_CFC_vs_'+task+'_'+single_version_code+'.svg'
        )
        plot_simple_regression_parameters_for_one_dataset(
            merged_data=merged_data_single_version,
            independent_variable=independent_variable,
            filenames=[f'./Outputs/Figures/plot_regression_intercept_CFC_vs_'+task+'_'+single_version_code+'.svg',
                        './Outputs/Figures/plot_regression_slope_CFC_vs_'+task+'_'+single_version_code+'.svg',
                        './Outputs/Figures/plot_regression_piechart_CFC_vs_'+task+'_'+single_version_code+'.svg']
        )
        
del dataset, merged_data_single_version, single_version_code, independent_variables
del requested_subset, dataframes_reduced, version_code
del merged_data







#%% plot_regression_of_LT_accuracy_against_Transfer_accuracy
requested_subsets = ('versions_equal_difficulty_across_gain_loss','versions_asymmetric_difficulty_across_gain_loss')

for requested_subset in requested_subsets:
    # prep data - filter to only include datasets with equal difficulty across gain and loss
    dataframes_reduced = filter_experiment_version(dataframes.copy(), requested_subset)
    version_code = get_version_code(requested_subset)
    [merged_data] = prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets(dataframes_reduced['LearningTask'], 
                                                                                            dataframes_reduced['SymbolChoice'], 
                                                                                            dataframes_reduced['PairChoice'], 
                                                                                            dataframes_reduced['Demographics'])
    # check that variables are not between 0 and 1
    assert merged_data['LT_correct_gain_minus_loss'].max() > 1.5, f"Error: {'LT_correct_gain_minus_loss'} values look like they are still coded 0-1" 
    assert merged_data['SC_correct_gain_minus_loss'].max() > 1.5, f"Error: {'SC_correct_gain_minus_loss'} values look like they are still coded 0-1" 

    plot_regression_of_LT_accuracy_against_Transfer_accuracy(merged_data=merged_data,
                                                            filenames=[f'./Outputs/Figures/regression_of_LT_accuracy_against_Transfer_accuracy_{version_code}.svg',
                                                                        f'./Outputs/Figures/regression_of_LT_accuracy_against_Transfer_accuracy_intercept_{version_code}.svg']
                                                            )





#%%  ANOVA results: effect of difficulty manipulation x CFC treatment on chose gain pair in CFC in versions_equal_and_asymmetric_difficulty_click_desired_1

# prep data - filter to only include datasets with equal difficulty across gain and loss
requested_subset = 'versions_equal_and_asymmetric_difficulty_click_desired_1'
dataframes_reduced = filter_experiment_version(dataframes.copy(), requested_subset)
version_code = get_version_code(requested_subset)
[merged_data] = prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets(dataframes_reduced['LearningTask'], 
                                                                                        dataframes_reduced['SymbolChoice'], 
                                                                                        dataframes_reduced['PairChoice'], 
                                                                                        dataframes_reduced['Demographics'])
# check that variables are not between 0 and 1
assert merged_data["CFC_chose_highest_expected_value"].max() > 1.5, "Error: CFC_chose_highest_expected_value values look like they are still coded 0-1"
assert merged_data['LT_correct_gain_minus_loss'].max() > 1.5, f"Error: {'LT_correct_gain_minus_loss'} values look like they are still coded 0-1" 
assert merged_data['SC_correct_gain_minus_loss'].max() > 1.5, f"Error: {'SC_correct_gain_minus_loss'} values look like they are still coded 0-1"    

# run ANOVA
ANOVA_effect_of_identify_best_and_asymmetric_difficulty(merged_df = merged_data)




