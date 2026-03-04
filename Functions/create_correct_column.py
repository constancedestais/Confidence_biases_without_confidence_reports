import numpy as np
import pandas as pd

def create_correct_column(df,task_name):

    '''
    INPUTS: 
    - df with exp_ID and chose_highest_expected_value columns
    - task_name ('LearningTask', 'PairChoice', 'SymbolChoice')

    OUTPUTS:
    - df with correct column added
    '''

    df['correct'] = None

    if (task_name == 'LearningTask'):
        # Define which experiment versions should have the original value (aka where choosing correctly is the the same as selecting the option with the highest expected value)
        # and which should have the opposite value
        # Replace these lists with your actual experiment versions
        original_value_versions = ['cd1_2025_click_desired_1_identify_best_1', 'cd1_2025_click_desired_1_identify_best_0']
        opposite_value_versions = ['cd1_2025_click_desired_0_identify_best_1', 'cd1_2025_click_desired_0_identify_best_0']

    elif (task_name in ['PairChoice','SymbolChoice']):
        # Define which experiment versions should have the original value (aka where choosing correctly is the the same as selecting the option with the highest expected value)
        # and which should have the opposite value
        # Replace these lists with your actual experiment versions
        original_value_versions = ['cd1_2025_click_desired_1_identify_best_1', 'cd1_2025_click_desired_0_identify_best_1']
        opposite_value_versions = ['cd1_2025_click_desired_1_identify_best_0', 'cd1_2025_click_desired_0_identify_best_0']
    else:
        raise ValueError("task_name must be 'LearningTask', 'PairChoice' or 'SymbolChoice'")


    # Set 'correct' for versions that should have the same value 
    for version_base in original_value_versions:
        mask = df['exp_ID'].str.contains(version_base, regex=False)
        df.loc[mask, 'correct'] = df.loc[mask, 'chose_highest_expected_value']
    del version_base
    # Set 'correct' for versions that should have the opposite value
    for version_base in opposite_value_versions:
        mask = df['exp_ID'].str.contains(version_base, regex=False)
        df.loc[mask, 'correct'] = 1 - df.loc[mask, 'chose_highest_expected_value']
    
    return df
