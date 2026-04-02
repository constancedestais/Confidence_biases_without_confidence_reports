import pandas as pd
import numpy as np
from scipy import stats
from Functions.scale_0_1_to_0_100  import scale_0_1_to_0_100

def prepare_data_averaged_by_valence_for_figures_comparing_multiple_datasets(LearningTask, SymbolChoice, PairChoice):
    # ----------- GOAL: average values per participant, trial and valence -----------
    '''
    GOAL: want dataset with one row per participant, with columns for:
        - participant_ID
        - pCorrect gain in LT -> with correct coded as percentage (0-100) not proportion (0-1) 
        - pCorrect loss in LT -> with correct coded as percentage (0-100) not proportion (0-1) 
        - pCorrect[gain-loss] in LT -> with correct coded as percentage (0-100) not proportion (0-1) 
        - pCorrect gain in SC -> with correct coded as percentage (0-100) not proportion (0-1) 
        - pCorrect loss in SC -> with correct coded as percentage (0-100) not proportion (0-1) 
        - pCorrect[gain-loss] in SC -> with correct coded as percentage (0-100) not proportion (0-1) 
        - pChoseGain in CFC -> with correct coded as percentage (0-100) not proportion (0-1) 
        - info about experiment version : exp_ID, LT_unequal_difficulty_binary, identify_best, click_desired
    '''

    # copy dataframes to avoid modifying originals
    LT = LearningTask.copy()
    SC = SymbolChoice.copy()
    CFC = PairChoice.copy()

    # remove practice session
    LT = LT[(LT.session != 0)]

    # ------- prepare Learning Task -------
    # get mean correct per participant, trial type (gain vs loss) - keep columns about exp version and LT_unequal_difficulty_binary for later analyses
    LT_mean_per_participant_per_valence = (LearningTask
        .assign(trial_type=LearningTask["is_gain_trial"].map({True: "gain", False: "loss"}))
        .groupby(["exp_ID", "LT_unequal_difficulty_binary","click_desired", "participant_ID", "trial_type"], as_index=False)["correct"]
        .mean()
    )
    # scale to percentage
    LT_mean_per_participant_per_valence["correct"] = scale_0_1_to_0_100(LT_mean_per_participant_per_valence["correct"], label="LT correct")
    # get seperate columns for mean correct in gain vs loss 
    LT_mean_per_participant_per_valence_wide = (LT_mean_per_participant_per_valence
        .pivot_table(index=["exp_ID", "LT_unequal_difficulty_binary", "click_desired", "participant_ID"], columns="trial_type", values="correct")
        .rename(columns={"gain": "LT_correct_gain", "loss": "LT_correct_loss"})
        .reset_index()
    )
    # calculate difference score for gain vs loss in LT
    LT_mean_per_participant_per_valence_wide["LT_correct_gain_minus_loss"] = LT_mean_per_participant_per_valence_wide["LT_correct_gain"] - LT_mean_per_participant_per_valence_wide["LT_correct_loss"]

    # ------- prepare Symbol Choice Task -------
    SC_mean_per_participant_per_valence = (SymbolChoice
        .assign(trial_type=SymbolChoice["is_gain_trial"].map({True: "gain", False: "loss"}))
        .groupby(["participant_ID", "trial_type"], as_index=False)["correct"]
        .mean()
    )
    # scale to percentage
    SC_mean_per_participant_per_valence["correct"] = scale_0_1_to_0_100(SC_mean_per_participant_per_valence["correct"], label="SC correct")
    # get seperate columns for mean correct in gain vs loss 
    SC_mean_per_participant_per_valence_wide = (SC_mean_per_participant_per_valence
        .pivot_table(index="participant_ID", columns="trial_type", values="correct")
        .rename(columns={"gain": "SC_correct_gain", "loss": "SC_correct_loss"})
        .reset_index()
    )
    # calculate difference score for gain vs loss in SC
    SC_mean_per_participant_per_valence_wide["SC_correct_gain_minus_loss"] = SC_mean_per_participant_per_valence_wide["SC_correct_gain"] - SC_mean_per_participant_per_valence_wide["SC_correct_loss"]

    # ------- prepare CFC -------
    CFC_mean_per_participant = (CFC
        .groupby(["participant_ID", "identify_best"], as_index=False)["chose_highest_expected_value"]
        .mean()
    )
    # scale to percentage
    CFC_mean_per_participant["chose_highest_expected_value"] = scale_0_1_to_0_100(CFC_mean_per_participant["chose_highest_expected_value"], label="chose_highest_expected_value")
    # rename column
    CFC_mean_per_participant = CFC_mean_per_participant.rename(columns={"chose_highest_expected_value": "CFC_chose_highest_expected_value"})

    # ------- prepare CFC -------

    # merge all four dataframes
    merged_df = pd.merge(CFC_mean_per_participant, LT_mean_per_participant_per_valence_wide, on=['participant_ID'])
    merged_df = pd.merge(merged_df, SC_mean_per_participant_per_valence_wide, on=['participant_ID'])

    return [merged_df]




def prepare_CFC_data_by_pair_composition_for_figures_comparing_multiple_datasets(PairChoice):

    # ----------- GOAL: in CFC data, average values per participant per pair type (new/not new vs homogenous/hetereogenous) -----------
    '''
    Docstring for prepare_CFC_data_by_pair_composition_for_figures_comparing_multiple_datasets
    INPUT: PairChoice dataframe with columns including:
        - participant_ID
        - chose_highest_expected_value (coded as 0-1)
        - n_new_pairs (coded as 0,1,2)
        - includes_new_pair (coded as 0,1 or False, True)
        - pair_valence_composition (coded as 'heterogeneous_symbol_valence' or 'homogeneous_symbol_valence')
    OUTPUT: four dataframes with one row per participant and columns for:
        1) CFC_mean_per_participant_by_n_new_pair:
            participant_ID, 
            n_new_pairs, 
            chose_highest_expected_value - averaged across trials for each participant and n_new_pairs condition; coded as percentage 0-100 not proportion 0-1
        2) CFC_mean_per_participant_by_pair_valence_composition:
            participant_ID, 
            pair_valence_composition, 
            chose_highest_expected_value - averaged across trials for each participant and pair_valence_composition condition; coded as percentage 0-100 not proportion 0-1
        3) CFC_mean_per_participant_by_includes_new_pair:
            participant_ID, 
            includes_new_pair, 
            chose_highest_expected_value - averaged across trials for each participant and includes_new_pair condition; coded as percentage 0-100 not proportion 0-1
        4) CFC_mean_per_participant_by_pair_valence_composition_and_includes_new_pair:
            participant_ID, 
            pair_valence_composition, 
            includes_new_pair,  
            chose_highest_expected_value - averaged across trials for each participant and combination of pair_valence_composition and includes_new_pair condition; coded as percentage 0-100 not proportion 0-1
            
    '''

    CFC = PairChoice.copy()

    # average pChoseGain by values n_new_pairs (0,1,2)
    CFC_mean_per_participant_by_n_new_pair = CFC.groupby(['participant_ID','n_new_pairs'])[['chose_highest_expected_value']].mean()
    CFC_mean_per_participant_by_n_new_pair = CFC_mean_per_participant_by_n_new_pair.reset_index()
    # scale to percentage
    CFC_mean_per_participant_by_n_new_pair["chose_highest_expected_value"] = scale_0_1_to_0_100(CFC_mean_per_participant_by_n_new_pair["chose_highest_expected_value"], label="CFC chose highest expected value")

    # average pChoseGain by includes_new_pair (0,1)
    # check that column "includes_new_pair" exists
    assert "includes_new_pair" in CFC.columns, "Column 'includes_new_pair' not found in CFC dataframe. Please create this column before running this function."
    CFC_mean_per_participant_by_includes_new_pair = CFC.groupby(['participant_ID','includes_new_pair'])[['chose_highest_expected_value']].mean()
    CFC_mean_per_participant_by_includes_new_pair = CFC_mean_per_participant_by_includes_new_pair.reset_index()   
    # scale to percentage
    CFC_mean_per_participant_by_includes_new_pair["chose_highest_expected_value"] = scale_0_1_to_0_100(CFC_mean_per_participant_by_includes_new_pair["chose_highest_expected_value"], label="CFC chose highest expected value")

    # average pChoseGain by pair_valence_composition (heterogeneous_symbol_valence, homogeneous_symbol_valence)
    CFC_mean_per_participant_by_pair_valence_composition = CFC.groupby(['participant_ID','pair_valence_composition'])[['chose_highest_expected_value']].mean()
    CFC_mean_per_participant_by_pair_valence_composition = CFC_mean_per_participant_by_pair_valence_composition.reset_index()
    # scale to percentage
    CFC_mean_per_participant_by_pair_valence_composition["chose_highest_expected_value"] = scale_0_1_to_0_100(CFC_mean_per_participant_by_pair_valence_composition["chose_highest_expected_value"], label="CFC chose highest expected value")

    # sanity check: get unique values of pair_valence_composition for each valye of includes_new_pair to check that we have all combinations
    # when includes_new_pair == False, we should only have homogeneous pairs (homogeneous_symbol_valence)
    # when includes_new_pair == True, we should have both heterogeneous and homogeneous pairs (heterogeneous_symbol_valence and homogeneous_symbol_valence)
    unique_combinations = CFC.groupby('includes_new_pair')['pair_valence_composition'].unique()
    assert (unique_combinations[False] == ['homogeneous_symbol_valence']).all(), f"Error: when includes_new_pair == False, we should only have homogeneous pairs, but we have: {unique_combinations[False]}"
    assert set(unique_combinations[True]) == set(['heterogeneous_symbol_valence', 'homogeneous_symbol_valence']), f"Error: when includes_new_pair == True, we should have both heterogeneous and homogeneous pairs, but we have: {unique_combinations[True]}"

    # average pChoseGain by a combination of includes_new_pair and pair_valence_composition
    CFC_mean_per_participant_by_pair_valence_composition_and_includes_new_pair = CFC.groupby(['exp_ID','participant_ID','pair_valence_composition','includes_new_pair'])[['chose_highest_expected_value']].mean()
    CFC_mean_per_participant_by_pair_valence_composition_and_includes_new_pair = CFC_mean_per_participant_by_pair_valence_composition_and_includes_new_pair.reset_index()
    # scale to percentage
    CFC_mean_per_participant_by_pair_valence_composition_and_includes_new_pair["chose_highest_expected_value"] = scale_0_1_to_0_100(CFC_mean_per_participant_by_pair_valence_composition_and_includes_new_pair["chose_highest_expected_value"], label="CFC chose highest expected value")

    return [CFC_mean_per_participant_by_n_new_pair,
            CFC_mean_per_participant_by_pair_valence_composition,
            CFC_mean_per_participant_by_includes_new_pair,
            CFC_mean_per_participant_by_pair_valence_composition_and_includes_new_pair]