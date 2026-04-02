
## print stats for paper
from scipy import stats
import statsmodels.formula.api as smf
import numpy as np
from Functions import get_version_code
from Functions.filter_experiment_version import filter_experiment_version
from Functions.set_up_helpers import project_paths_for_main, load_multiple_csvs
from Functions.prepare_data_for_figures_comparing_multiple_datasets import (
    prepare_CFC_data_by_pair_composition_for_figures_comparing_multiple_datasets,
    prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets,
)
from Functions.ANOVAs_chose_gain_pair import (
    ANOVA_effect_of_click_desired_and_identify_best,
    ANOVA_effect_of_identify_best_and_asymmetric_difficulty,
)
from Functions.scale_0_1_to_0_100 import scale_0_1_to_0_100
from Functions.stats_helpers import run_one_sample_ttest_and_format, run_paired_ttest_and_format, run_linear_model_and_print


# =============================================================================
# Set up: import packages, set directories & load data
# =============================================================================

# Anchor all paths to THIS runner (project root)
directories = project_paths_for_main(__file__)

CSV_FILENAMES: dict[str, str] = {
    "LearningTask": "LearningTask.csv",
    "PairChoice": "CD1_PairChoice.csv",
    "SymbolChoice": "CD1_SymbolChoice.csv",
    "BonusRound": "CD1_BonusRound.csv",
    "Demographics": "CD1_Demographics.csv",
    "General": "CD1_General.csv",
}

dataframes = load_multiple_csvs(directories.data, CSV_FILENAMES)


# =============================================================================
# STATS FOR EFFECT OF PAIR COMPOSITION ON CHOICE IN CFC
# =============================================================================

print("\n ==================== STATS FOR EFFECT OF PAIR COMPOSITION ON CHOICE IN CFC (first in datasets with equal difficulty, then asymmetric difficulty) ==================== \n")
requested_subsets = ['versions_equal_difficulty_across_gain_loss', 'versions_asymmetric_difficulty_across_gain_loss']

for requested_subset in requested_subsets:
    print(f"\n =============  requested_subset: {requested_subset} ============= ")

    dataframes_reduced = filter_experiment_version(dataframes.copy(), requested_subset)

    [unused,
    CFC_mean_per_participant_by_pair_valence_composition,
    CFC_mean_per_participant_by_includes_new_pair,
    unused] = prepare_CFC_data_by_pair_composition_for_figures_comparing_multiple_datasets(dataframes_reduced['PairChoice'])
    assert CFC_mean_per_participant_by_includes_new_pair['chose_highest_expected_value'].max() > 1.5, f"Error: chose_highest_expected_value values look like they are still coded 0-1" 
    assert set(CFC_mean_per_participant_by_includes_new_pair['includes_new_pair'].unique()) == {True, False}, "Error: includes_new_pair should be boolean with values True and False"


    # ---------- effect of new vs old pair
    # create seperate columns for new and old pair
    CFC_mean_per_participant_by_includes_new_pair_wide = CFC_mean_per_participant_by_includes_new_pair.pivot(index="participant_ID", columns="includes_new_pair", values="chose_highest_expected_value").reset_index()
    # rename columns for clarity
    CFC_mean_per_participant_by_includes_new_pair_wide = CFC_mean_per_participant_by_includes_new_pair_wide.rename(columns={False: "old_pair", True: "new_pair"})
    # one sample t-test against 50 for each condition
    for pair_type in ["old_pair", "new_pair"]:
        run_one_sample_ttest_and_format(
            CFC_mean_per_participant_by_includes_new_pair_wide[pair_type],
            population_mean_under_null=50,
            label=f"CFC - overall p_chose_highest_EV vs 50 for {pair_type}",
        )
    # paired t-test comparing new vs old pair
    paired_cohens_d_new_vs_old = run_paired_ttest_and_format(
        CFC_mean_per_participant_by_includes_new_pair_wide["new_pair"],
        CFC_mean_per_participant_by_includes_new_pair_wide["old_pair"],
        label="CFC - p_chose_highest_EV (new vs old pair)",
    )

    # ---------- effect of heterogenous valence vs homoegeneous valence in pair
    # create seperate columns for heterogenous and homogeneous valence pairs
    CFC_mean_per_participant_by_pair_valence_composition_wide = CFC_mean_per_participant_by_pair_valence_composition.pivot(index="participant_ID", columns="pair_valence_composition", values="chose_highest_expected_value").reset_index()
    # rename columns for clarity
    CFC_mean_per_participant_by_pair_valence_composition_wide = CFC_mean_per_participant_by_pair_valence_composition_wide.rename(columns={"heterogeneous_symbol_valence": "heterogenous_valence_pair", "homogeneous_symbol_valence": "homogeneous_valence_pair"})
    # one sample t-test against 50 for each condition
    for pair_type in ["heterogenous_valence_pair", "homogeneous_valence_pair"]:
        run_one_sample_ttest_and_format(
            CFC_mean_per_participant_by_pair_valence_composition_wide[pair_type],
            population_mean_under_null=50,
            label=f"CFC - overall p_chose_highest_EV vs 50 for {pair_type}",
        )
    # paired t-test comparing heterogenous vs homogeneous valence pairs
    paired_cohens_d_heterogenous_vs_homogeneous = run_paired_ttest_and_format(
        CFC_mean_per_participant_by_pair_valence_composition_wide["heterogenous_valence_pair"],
        CFC_mean_per_participant_by_pair_valence_composition_wide["homogeneous_valence_pair"],
        label="CFC - p_chose_highest_EV (heterogenous vs homogeneous valence pair)",
    )   

    # ---------- repeat winning regression only in data with new pairs
    # first have to get merged data
    # filted CFC data to only include new pairs
    CFC_new_pairs = dataframes_reduced['PairChoice'][dataframes_reduced['PairChoice']["includes_new_pair"] == True].copy()
    # prepare merged data with only new pairs in CFC
    [ merged_data_new_pairs] = prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets(
        dataframes_reduced['LearningTask'],
        dataframes_reduced['SymbolChoice'],
        CFC_new_pairs,
        dataframes_reduced['Demographics'],
    )
    # check that dependent and independent variables is not between 0 and 1
    assert merged_data_new_pairs["CFC_chose_highest_expected_value"].max() > 1.5, "Error: CFC_chose_highest_expected_value values look like they are still coded 0-1"
    assert merged_data_new_pairs["LT_correct_gain_minus_loss"].max() > 1.5, f"Error: LT_correct_gain_minus_loss values look like they are still coded 0-1" 
    assert merged_data_new_pairs["SC_correct_gain_minus_loss"].max() > 1.5, f"Error: SC_correct_gain_minus_loss values look like they are still coded 0-1" 
    # regression stats (p_chose_highest_EV ~ accuracy(gain-loss) + identify_best)
    independent_variables_gain_minus_loss = ["LT_correct_gain_minus_loss", "SC_correct_gain_minus_loss"]
    for independent_variable_name in independent_variables_gain_minus_loss:
        # check that dependent and independent variables is not between 0 and 1
        assert merged_data_new_pairs["CFC_chose_highest_expected_value"].max() > 1.5, "Error: CFC_chose_highest_expected_value values look like they are still coded 0-1"
        assert merged_data_new_pairs[independent_variable_name].max() > 1.5, f"Error: {independent_variable_name} values look like they are still coded 0-1" 
        # run OLS regression 
        run_linear_model_and_print(
            dataframe=merged_data_new_pairs,
            dependent_variable_name="CFC_chose_highest_expected_value",
            independent_variable_names=[independent_variable_name],
            categorical_variable_names=["identify_best"],
            label_prefix="",
        )




# =============================================================================
# STATS FOR EACH INDIVIDUAL DATASET
# =============================================================================

all_exp_versions = [
    "cd1_2025_click_desired_1_identify_best_1",
    "cd1_2025_click_desired_1_identify_best_0",
    "cd1_2025_click_desired_0_identify_best_1",
    "cd1_2025_click_desired_0_identify_best_0",
    "cd1_2025_click_desired_1_identify_best_1_difficulty_0_70_0_80",
    "cd1_2025_click_desired_1_identify_best_0_difficulty_0_70_0_80",
]
assert type(all_exp_versions) == list, "Error: exp_versions should be a list (to do this use square brackets [])"

for exp_version in all_exp_versions:
    print(f"\n ==================== STATS FOR INDIVIDUAL DATASET: {exp_version} ==================== \n")

    dataframes_copy = dataframes.copy()
    single_version_data = filter_experiment_version(dataframes_copy, exp_version)
    # check df are not empty
    assert not (single_version_data["LearningTask"].empty), "Problem: dataframes in single_version_data are empty"

    LearningTask = single_version_data["LearningTask"].copy()
    SymbolChoice = single_version_data["SymbolChoice"].copy()
    CFC = single_version_data["PairChoice"].copy()
    Demographics = single_version_data["Demographics"].copy()

    # ------ demographics stats ------
    n_participants = single_version_data["LearningTask"]["participant_ID"].nunique()
    print(f"\nN_participants = {n_participants}")
    n_female = single_version_data["Demographics"]["gender"].value_counts().get("Female", 0)
    n_male = single_version_data["Demographics"]["gender"].value_counts().get("Male", 0)
    n_other = single_version_data["Demographics"]["gender"].value_counts().get("Other", 0)
    missing_data = n_participants - (n_female + n_male + n_other)
    print(
        f"\nGender breakdown: female = {n_female}; male = {n_male}; other = {n_other}; "
        f"missing_data = {missing_data} (NB: 2 participants in v01 did not complete Demographics/General);"
    )
    mean_age = single_version_data["Demographics"]["age"].mean()
    std_age = single_version_data["Demographics"]["age"].std()
    print(f"\nage = {mean_age:.2f}±{std_age:.2f} years;")

    # ------ ttest Choice accuracy compared to chance, overall in the Learning Task ------
    # get mean correct per participant
    LT_mean_per_participant = (LearningTask
            .groupby(["participant_ID"], as_index=False)["correct"]
            .mean())
    # scale by 100
    LT_mean_per_participant["correct"] = scale_0_1_to_0_100( LT_mean_per_participant["correct"], "LT_mean_per_participant['correct']")
    assert LT_mean_per_participant["correct"].max() > 1.5, "Error: overall mean correct values look like they are still coded 0-1"
    # one-sample t-test against 50
    run_one_sample_ttest_and_format(
        LT_mean_per_participant["correct"],
        population_mean_under_null=50,
        label="LT - overall accuracy vs 50",
    )

    # ------ ttest Choice accuracy compared to chance, overall in the Transfer Task ------
    # get mean correct per participant
    SC_mean_per_participant = (SymbolChoice
            .groupby(["participant_ID"], as_index=False)["correct"]
            .mean())
    # scale by 100
    SC_mean_per_participant["correct"] = scale_0_1_to_0_100( SC_mean_per_participant["correct"], "SC_mean_per_participant['correct']")
    assert SC_mean_per_participant["correct"].max() > 1.5, "Error: overall mean correct values look like they are still coded 0-1"
    # one-sample t-test against 50
    run_one_sample_ttest_and_format(
        SC_mean_per_participant["correct"],
        population_mean_under_null=50,
        label="SC - overall accuracy vs 50",
    )

    # ------ paired-ttest Choice accuracy in gains vs losses in the Learning Task ------
    # get mean correct per participant
    LT_mean_per_participant_per_valence = (LearningTask
            .assign(trial_type=LearningTask["is_gain_trial"].map({True: "gain", False: "loss"}))
            .groupby(["participant_ID", "trial_type"], as_index=False)["correct"]
            .mean())
    # scale by 100
    LT_mean_per_participant_per_valence["correct"] = scale_0_1_to_0_100( LT_mean_per_participant_per_valence["correct"], "LT_mean_per_participant_per_valence['correct']")
    # make separate columns for gain and loss mean correct
    LT_mean_per_participant_per_valence = LT_mean_per_participant_per_valence.pivot(index="participant_ID", columns="trial_type", values="correct").reset_index()
    # paired t-test
    paired_cohens_d_lt = run_paired_ttest_and_format(
        LT_mean_per_participant_per_valence["gain"],
        LT_mean_per_participant_per_valence["loss"],
        label="LT - accuracy(gain vs. loss)",
    )

    # ------ paired-ttest Choice accuracy in gains vs losses in the Transfer Task ------
    # get mean correct per participant 
    SC_mean_per_participant_per_valence = (SymbolChoice
            .assign(trial_type=SymbolChoice["is_gain_trial"].map({True: "gain", False: "loss"}))
            .groupby(["participant_ID", "trial_type"], as_index=False)["correct"]
            .mean())
    # scale by 100
    SC_mean_per_participant_per_valence["correct"] = scale_0_1_to_0_100( SC_mean_per_participant_per_valence["correct"], "SC_mean_per_participant_per_valence['correct']")
    # make separate columns for gain and loss mean correct
    SC_mean_per_participant_per_valence = SC_mean_per_participant_per_valence.pivot(index="participant_ID", columns="trial_type", values="correct").reset_index()
    # paired t-test
    paired_cohens_d_sc = run_paired_ttest_and_format(
        SC_mean_per_participant_per_valence["gain"],
        SC_mean_per_participant_per_valence["loss"],
        label="SC - accuracy(gain vs. loss)",
    )

    # ------ ttest p_chose_highest_EV in CFC ------
    # get mean chose_highest_expected_value per participant 
    CFC_mean_per_participant = (CFC
            .groupby(["participant_ID"], as_index=False)["chose_highest_expected_value"]
            .mean())
        # scale by 100
    CFC_mean_per_participant["chose_highest_expected_value"] = scale_0_1_to_0_100( CFC_mean_per_participant["chose_highest_expected_value"], "CFC_mean_per_participant['chose_highest_expected_value']")
    assert CFC_mean_per_participant["chose_highest_expected_value"].max() > 1.5, "Error: p_chose_highest_EV values look like they are still coded 0-1"
    # one-sample t-test against 50
    run_one_sample_ttest_and_format(
        CFC_mean_per_participant["chose_highest_expected_value"],
        population_mean_under_null=50,
        label="CFC - overall p_chose_highest_EV vs 50",
    )
    a=1

# =============================================================================
# REGRESSION FOR cd1_2025_click_desired_1_identify_best_1
# =============================================================================

print("\n ==================== REGRESSION FOR cd1_2025_click_desired_1_identify_best_1 ==================== \n")

v11 = "cd1_2025_click_desired_1_identify_best_1"
dataframes_copy = dataframes.copy()
v11_data = filter_experiment_version(dataframes_copy, v11)

# sanity check - get unique experiment versions in Learning Task, should be equal to v11
assert set(v11_data["LearningTask"]["exp_ID"].unique()) == {v11}, "Problem: wrong exp version in v11_data"

[merged_data ] = prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets(
    v11_data["LearningTask"],
    v11_data["SymbolChoice"],
    v11_data["PairChoice"],
    v11_data["Demographics"],
)

for independent_variable_name in ["LT_correct_gain_minus_loss", "SC_correct_gain_minus_loss"]:
    # check that independent variable is not between 0 and 1
    assert merged_data[independent_variable_name].max() > 1.5, f"Error: {independent_variable_name} values look like they are still coded 0-1"
    # run OLS regression with only this independent variable (no covariates) to predict CFC_chose_highest_expected_value
    run_linear_model_and_print(
        dataframe=merged_data,
        dependent_variable_name="CFC_chose_highest_expected_value",
        independent_variable_names=[independent_variable_name],
        categorical_variable_names=[],
        label_prefix="[v11] ",
    )


# =============================================================================
# Demographics stats
# =============================================================================

print("\n ==================== DEMOGRAPHICS STATS ACROSS ALL 6 VERSIONS ==================== \n")

version = "all"
dataframes_copy = dataframes.copy()
all_data = filter_experiment_version(dataframes_copy, version)

n_participants = all_data["LearningTask"]["participant_ID"].nunique()
print(f"\nN = {n_participants} although 2 participants in v01 did not complete Demographics/General;")

n_female = all_data["Demographics"]["gender"].value_counts().get("Female", 0)
n_male = all_data["Demographics"]["gender"].value_counts().get("Male", 0)
n_other = all_data["Demographics"]["gender"].value_counts().get("Other", 0)
print(f"\nGender breakdown: female = {n_female}; male = {n_male}; other = {n_other};")

mean_age = all_data["Demographics"]["age"].mean()
std_age = all_data["Demographics"]["age"].std()
print(f"\nage = {mean_age:.2f}±{std_age:.2f} years;")

# =============================================================================
# STATS FOR MULTIPLE DATASETS WITH EQUAL DIFFICULTY ACROSS GAIN AND LOSS
# =============================================================================

print("\n ==================== STATS FOR MULTIPLE DATASETS WITH EQUAL DIFFICULTY ACROSS GAIN AND LOSS ==================== \n")

# load and prep data
first_four_versions = "versions_equal_difficulty_across_gain_loss"
dataframes_copy = dataframes.copy()
first_four_versions_data = filter_experiment_version(dataframes_copy, first_four_versions)

# sanity check - get unique experiment versions in Learning Task
unique_exp_versions = first_four_versions_data["LearningTask"]["exp_ID"].unique()
print("\n unique experiment versions: \n", unique_exp_versions)

# prepare LT/SC/CFC merged data
[ merged_data] = prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets(
    first_four_versions_data["LearningTask"],
    first_four_versions_data["SymbolChoice"],
    first_four_versions_data["PairChoice"],
    first_four_versions_data["Demographics"],
)

# 1. ANOVA stats: effect of click_desired_0/click_desired_1 x identify_best_0/identify_best_1 on p_chose_highest_EV
# check that dependent variable is not between 0 and 1
assert merged_data["CFC_chose_highest_expected_value"].max() > 1.5, "Error: CFC_chose_highest_expected_value values look like they are still coded 0-1"
# run ANOVA
ANOVA_effect_of_click_desired_and_identify_best(merged_df=merged_data)

# 2. regression stats (p_chose_highest_EV ~ accuracy(gain-loss) + identify_best)
independent_variables_gain_minus_loss = ["LT_correct_gain_minus_loss", "SC_correct_gain_minus_loss"]
for independent_variable_name in independent_variables_gain_minus_loss:
    # check that dependent and independent variables is not between 0 and 1
    assert merged_data["CFC_chose_highest_expected_value"].max() > 1.5, "Error: CFC_chose_highest_expected_value values look like they are still coded 0-1"
    assert merged_data[independent_variable_name].max() > 1.5, f"Error: {independent_variable_name} values look like they are still coded 0-1" 
    # run OLS regression 
    run_linear_model_and_print(
        dataframe=merged_data,
        dependent_variable_name="CFC_chose_highest_expected_value",
        independent_variable_names=[independent_variable_name],
        categorical_variable_names=["identify_best"],
        label_prefix="",
    )

# 3. behavioural tests over combined datasets
print("----------------- behavioural ttests over combined datasets (versions_equal_difficulty_across_gain_loss) -----------------")

LearningTask = first_four_versions_data["LearningTask"].copy()
SymbolChoice = first_four_versions_data["SymbolChoice"].copy()
CFC          = first_four_versions_data["PairChoice"].copy()
Demographics = first_four_versions_data["Demographics"].copy()

# ------ paired-ttest Choice accuracy in gains vs losses in the Learning Task ------
# get mean correct per participant
LT_mean_per_participant_per_valence = (LearningTask
        .assign(trial_type=LearningTask["is_gain_trial"].map({True: "gain", False: "loss"}))
        .groupby(["participant_ID", "trial_type"], as_index=False)["correct"]
        .mean())
# scale by 100
LT_mean_per_participant_per_valence["correct"] = scale_0_1_to_0_100( LT_mean_per_participant_per_valence["correct"], "LT_mean_per_participant_per_valence['correct']")
# make separate columns for gain and loss mean correct
LT_mean_per_participant_per_valence = LT_mean_per_participant_per_valence.pivot(index="participant_ID", columns="trial_type", values="correct").reset_index()
# paired t-test
paired_cohens_d_lt = run_paired_ttest_and_format(
    LT_mean_per_participant_per_valence["gain"],
    LT_mean_per_participant_per_valence["loss"],
    label="LT - accuracy(gain vs. loss)",
)

# ------ paired-ttest Choice accuracy in gains vs losses in the Transfer Task ------
# get mean correct per participant 
SC_mean_per_participant_per_valence = (SymbolChoice
        .assign(trial_type=SymbolChoice["is_gain_trial"].map({True: "gain", False: "loss"}))
        .groupby(["participant_ID", "trial_type"], as_index=False)["correct"]
        .mean())
# scale by 100
SC_mean_per_participant_per_valence["correct"] = scale_0_1_to_0_100( SC_mean_per_participant_per_valence["correct"], "SC_mean_per_participant_per_valence['correct']")
# make separate columns for gain and loss mean correct
SC_mean_per_participant_per_valence = SC_mean_per_participant_per_valence.pivot(index="participant_ID", columns="trial_type", values="correct").reset_index()
# paired t-test
paired_cohens_d_sc = run_paired_ttest_and_format(
    SC_mean_per_participant_per_valence["gain"],
    SC_mean_per_participant_per_valence["loss"],
    label="SC - accuracy(gain vs. loss)",
)

# ------ ttest p_chose_highest_EV in CFC ------
# get mean chose_highest_expected_value per participant 
CFC_mean_per_participant = (CFC
        .groupby(["participant_ID"], as_index=False)["chose_highest_expected_value"]
        .mean())
    # scale by 100
CFC_mean_per_participant["chose_highest_expected_value"] = scale_0_1_to_0_100( CFC_mean_per_participant["chose_highest_expected_value"], "CFC_mean_per_participant['chose_highest_expected_value']")
assert CFC_mean_per_participant["chose_highest_expected_value"].max() > 1.5, "Error: p_chose_highest_EV values look like they are still coded 0-1"
# one-sample t-test against 50
run_one_sample_ttest_and_format(
    CFC_mean_per_participant["chose_highest_expected_value"],
    population_mean_under_null=50,
    label="CFC - overall p_chose_highest_EV vs 50",
)
a=1


# =============================================================================
# STATS FOR MULTIPLE DATASETS WITH RL+ BUT BOTH EQUAL AND ASYMMETRIC DIFFICULTY
# =============================================================================

print("\n ==================== STATS FOR MULTIPLE DATASETS WITH RL+ BUT BOTH EQUAL AND ASYMMETRIC DIFFICULTY ==================== \n")

equal_and_asymmetric = "versions_equal_and_asymmetric_difficulty_click_desired_1"
dataframes_copy = dataframes.copy()
equal_and_asymmetric_data = filter_experiment_version(dataframes_copy, equal_and_asymmetric)

# sanity check - get unique experiment versions in Learning Task
unique_exp_versions = equal_and_asymmetric_data["LearningTask"]["exp_ID"].unique()
print("\nunique experiment versions: \n", unique_exp_versions)

[merged_data] = prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets(
    equal_and_asymmetric_data["LearningTask"],
    equal_and_asymmetric_data["SymbolChoice"],
    equal_and_asymmetric_data["PairChoice"],
    equal_and_asymmetric_data["Demographics"],
)

# 1. ANOVA stats: effect of difficulty(equal/asymmetric) x CFC(identify_best_1/identify_best_0) on p_chose_highest_EV
ANOVA_effect_of_identify_best_and_asymmetric_difficulty(merged_df=merged_data)


# =============================================================================
# STATS FOR TWO DATASETS WITH ASYMMETRIC DIFFICULTY
# =============================================================================

print("\n ==================== STATS FOR TWO DATASETS WITH ASYMMETRIC DIFFICULTY ==================== \n")

asymmetric = "versions_asymmetric_difficulty_across_gain_loss"
dataframes_copy = dataframes.copy()
asymmetric_data = filter_experiment_version(dataframes_copy, asymmetric)

# sanity check - get unique experiment versions in Learning Task
unique_exp_versions = asymmetric_data["LearningTask"]["exp_ID"].unique()
print("\nunique experiment versions: \n", unique_exp_versions)

[merged_data] = prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets(
    asymmetric_data["LearningTask"],
    asymmetric_data["SymbolChoice"],
    asymmetric_data["PairChoice"],
    asymmetric_data["Demographics"],
)

# regression stats (p_chose_highest_EV ~ accuracy(gain-loss) + identify_best)
for independent_variable_name in ["LT_correct_gain_minus_loss", "SC_correct_gain_minus_loss"]:
    # check that dependent and independent variables is not between 0 and 1
    assert merged_data["CFC_chose_highest_expected_value"].max() > 1.5, "Error: CFC_chose_highest_expected_value values look like they are still coded 0-1"
    assert merged_data[independent_variable_name].max() > 1.5, f"Error: {independent_variable_name} values look like they are still coded 0-1" 
    # run OLS regression
    run_linear_model_and_print(
        dataframe=merged_data,
        dependent_variable_name="CFC_chose_highest_expected_value",
        independent_variable_names=[independent_variable_name],
        categorical_variable_names=["identify_best"],
        label_prefix="",
    )











