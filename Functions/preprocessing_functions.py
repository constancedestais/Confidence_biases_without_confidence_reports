import re
import pandas as pd

def convert_ID_columns_to_string(df):
    df['prolific_ID'] = df['prolific_ID'].apply(lambda x: str(x) if not isinstance(x, str) else x)
    df['manual_ID'] = df['manual_ID'].apply(lambda x: str(x) if not isinstance(x, str) else x)
    df['session_ID'] = df['session_ID'].apply(lambda x: str(x) if not isinstance(x, str) else x)
    return df

def get_pair_key(row):
    '''
    Create a standardized representation of each pair (sort the values) in a row
    '''
    # Sort the two symbol IDs to ensure order doesn't matter
    symbols = sorted([row['symbol_chosen_id'], row['symbol_unchosen_id']])
    return tuple(symbols)

def create_pair_number_column_LearningTask(df):
    '''
    create column for pair number (1-4) based on unique symbol combination within each block --> pairs 1 and 2 are GAIN, pairs 3 and 4 are LOSS 
    '''
    # Apply the function to create a new column with the standardized pair (for each row, pandas automatically passes that row as an argument to get_pair_key())
    df['symbol_pair'] = df.apply(get_pair_key, axis=1)
    # Create a fixed mapping for the predefined pairs
    predefined_pairs = [ (0, 1), (2, 3), (4, 5), (6, 7)]
    # Create mapping dictionary
    pair_mapping = {pair: i for i, pair in enumerate(predefined_pairs)}
    # Create the pair_number column using the mapping
    df['pair_number'] = df['symbol_pair'].map(pair_mapping)
    # Handle any unexpected pairs (if there are any pairs not in our predefined list)
    assert ~(df['pair_number'].isna().any()), f"Error: Some rows in LearningTask contain combinations of pairs that were not in the predefined list"
    return df

def create_pair_number_column_SymbolChoice(df):
    '''
    create column for pair number (1-4) based on unique symbol combination within each block --> pairs 1 and 2 are GAIN, pairs 3 and 4 are LOSS 
    '''
    # Apply the function to create a new column with the standardized pair (for each row, pandas automatically passes that row as an argument to get_pair_key())
    df['symbol_pair'] = df.apply(get_pair_key, axis=1)
    # Create a fixed mapping for the predefined pairs
    predefined_pairs = [ (0, 1), (2, 3), (4, 5), (6, 7), 
                        (0, 3), (1, 2), (4, 7), (5, 6),
                        (0, 4), (2, 6), (0, 6), (2, 4),
                        (3, 7), (1, 5), (1, 7), (3, 5)]
    # Create mapping dictionary
    pair_mapping = {pair: i for i, pair in enumerate(predefined_pairs)}
    # Create the pair_number column using the mapping
    df['pair_number'] = df['symbol_pair'].map(pair_mapping)
    # Handle any unexpected pairs (if there are any pairs not in our predefined list)
    assert ~(df['pair_number'].isna().any()), f"Error: Some rows in SymbolChoice contain combinations of pairs that were not in the predefined list"
    return df

def get_combination_key(row):
    # Sort the two symbol IDs to ensure order doesn't matter - both order of symbol IDs in a pair and order of pairs
    pair1 = tuple(sorted([row['pair_chosen_symbol_top_id'], row['pair_chosen_symbol_bottom_id']]))
    pair2 = tuple(sorted([row['pair_unchosen_symbol_top_id'], row['pair_unchosen_symbol_bottom_id']]))
    # Sort the pairs themselves to ensure order doesn't matter between pairs
    # This will normalize ((4, 5), (0, 1)) to ((0, 1), (4, 5))
    pairs = sorted([pair1, pair2])
    combination = tuple(pairs)
    
    return combination

def create_pair_number_column_PairChoice(df):
    # Apply the function to create a new column with the standardized combination (for each row, pandas automatically passes that row as an argument to get_pair_key())
    df['pair_combination'] = df.apply(get_combination_key, axis=1)
    # Create a fixed mapping for the predefined pairs
    predefined_combinations = [
        ((0, 1), (4, 5)),
        ((0, 1), (6, 7)),
        ((0, 1), (4, 7)),
        ((0, 1), (5, 6)),

        ((2, 3), (4, 5)),
        ((2, 3), (6, 7)),
        ((2, 3), (4, 7)),
        ((2, 3), (5, 6)),

        ((0, 3), (4, 5)),
        ((0, 3), (6, 7)),
        ((0, 3), (4, 7)),
        ((0, 3), (5, 6)),

        ((1, 2), (4, 5)),
        ((1, 2), (6, 7)),
        ((1, 2), (4, 7)),
        ((1, 2), (5, 6)),

        ((0, 4), (3, 7)),
        ((0, 4), (1, 5)),
        ((0, 4), (1, 7)),
        ((0, 4), (3, 5)),

        ((2, 6), (3, 7)),
        ((1, 5), (2, 6)),
        ((1, 7), (2, 6)),
        ((2, 6), (3, 5)),

        ((0, 6), (3, 7)),
        ((0, 6), (1, 5)),
        ((0, 6), (1, 7)),
        ((0, 6), (3, 5)),

        ((2, 4), (3, 7)),
        ((1, 5), (2, 4)),
        ((1, 7), (2, 4)),
        ((2, 4), (3, 5))
    ]
    # Create mapping dictionary
    pair_mapping = {pair: i for i, pair in enumerate(predefined_combinations)}
    # Create the pair_number column using the mapping
    df['pair_number'] = df['pair_combination'].map(pair_mapping)
    # Handle any unexpected pairs (if there are any pairs not in our predefined list)
    assert not df['pair_number'].isna().any(), f"Error: Some rows in PairChoice contain combinations of pairs that were not in the predefined list"
    return df


def create_trial_per_pair_column(df):
    # re-number the trials, at the level of each pair, and each session, and each participant
    df = df.sort_values(['prolific_ID', 'session', 'pair_number', 'trial'])
    df['trial_per_pair'] = df.groupby(['prolific_ID', 'session', 'pair_number']).cumcount()
    return df

def create_participant_ID_column(df):
    '''
    Creates anonymized unique participant IDs that reflect each experiment version.
    
    For each unique experiment ID in the dataframe, this function:
    1. Extracts condition parameters from the experiment ID
    2. Creates sequential participant numbers for each unique prolific ID
    3. Generates participant IDs in the format "CD1_{click_desired}_{identify_best}_p{number}"
    
    INPUTS:
    -----------
    df : pandas.DataFrame
        Input dataframe containing 'exp_ID' and 'prolific_ID' columns
        
    OUTPUTS:
    --------
    pandas.DataFrame
        The modified dataframe with a new 'participant_ID' column
        
    Notes:
    ------
    - Assumes experiment IDs follow the format with 'click_desired_X_identify_best_Y'
    - Includes error handling for experiment IDs with different formats
    '''
    
    # Create a new dataframe to store the results
    result_df = df.copy()
    
    # Get unique experiment IDs
    unique_exp_ids = df['exp_ID'].unique()
    
    # Process each experiment ID separately
    for exp_id in unique_exp_ids:
        # Filter dataframe for current experiment
        exp_df = df[df['exp_ID'] == exp_id]
        
        # Extract condition values from this exp_ID
        match = re.search(r"click_desired_(\d+).*identify_best_(\d+)(?:_(.*))?", exp_id)
        if match:
            click_desired = int(match.group(1))
            identify_best = int(match.group(2))
            suffix        = match.group(3) or ""   # (or "" if nothing there)
        else:
            # Handle cases where the exp_ID format might be different
            raise KeyError(f"Warning: Could not parse exp_ID '{exp_id}'. Using default values.")
        
        '''
        try:
            click_desired, identify_best = exp_id.split('click_desired_')[1].split('_identify_best_')
            click_desired = int(click_desired)
            identify_best = int(identify_best)
        except (IndexError, ValueError) as e:
            # Handle cases where the exp_ID format might be different
            print(f"Warning: Could not parse exp_ID '{exp_id}'. Using default values.")
            click_desired = 0
            identify_best = 0
        '''

        # Get unique prolific_IDs for this experiment and assign sequential numbers
        unique_ids = exp_df['prolific_ID'].unique()
        id_map = {pid: f"CD1_v{click_desired}{identify_best}{'_' + suffix if suffix else ''}_p{i+1}" for i, pid in enumerate(unique_ids)}
        
        # Update the result dataframe with participant_IDs for this experiment
        for pid in unique_ids:
            mask = (result_df['exp_ID'] == exp_id) & (result_df['prolific_ID'] == pid)
            result_df.loc[mask, 'participant_ID'] = id_map[pid]
    
    return result_df

    ''''
    # Extract condition values from the single exp_ID
    click_desired, identify_best = df['exp_ID'].iloc[0].split('click_desired_')[1].split('_identify_best_')
    click_desired = int(click_desired)
    identify_best = int(identify_best)

    # Get unique prolific_IDs and assign a sequential number
    unique_ids = df['prolific_ID'].unique()
    id_map = {pid: f"CD1_{click_desired}_{identify_best}_p{i+1}" for i, pid in enumerate(unique_ids)}

    # Create the anonymized participant_ID column
    df['participant_ID'] = df['prolific_ID'].map(id_map)
    return df
    '''
    
def add_missing_participants(df, LearningTask, ID_and_info):
    """
    To `df`, add rows with NaN values for missing participants,
    but if the columns "identify_best", "exp_ID" and "click_desired" are present in the dataset,
    fill these in with the correct info for that participant from `ID_and_info`
    """
    # check necessary columns are present 
    assert 'prolific_ID' in df.columns, "Error: 'prolific_ID' column is missing from df dataframe"
    assert 'prolific_ID' in ID_and_info.columns, "Error: 'prolific_ID' column is missing from ID_and_info dataframe"
    assert 'exp_ID' in ID_and_info.columns, "Error: 'exp_ID' column is missing from ID_and_info dataframe"
    assert 'identify_best' in ID_and_info.columns, "Error: 'identify_best' column is missing from ID_and_info dataframe"
    assert 'click_desired' in ID_and_info.columns, "Error: 'click_desired' column is missing from ID_and_info dataframe"

    # identify missing participants
    missing_participants = LearningTask.loc[~LearningTask['prolific_ID'].isin(df['prolific_ID']), 'prolific_ID'].drop_duplicates()

    # create a new dataframe with rows for the missing participants with the same columns as the original dataframe
    pad = df.iloc[0:0].copy().reindex(range(len(missing_participants)))
    pad['prolific_ID'] = missing_participants.to_numpy() # transform column to array

    # if the following columns are present: exp_ID, identify_best, click_desired, in df, fill them in for the missing participants using info in ID_and_info dataframe
    for variable in ['exp_ID', 'identify_best', 'click_desired']:
        if variable in df.columns:
            # get the variable info for the missing participants from ID_and_info
            variable_info = ID_and_info[['prolific_ID', variable]].drop_duplicates()
            # fill in the variable column for the missing participants in pad using the variable info from ID_and_info, matching up participants by their participant IDs
            for participant in missing_participants:
                if participant not in variable_info['prolific_ID'].values:
                    raise ValueError(f"Error: Participant {participant} is missing from ID_and_info dataframe, so cannot fill in {variable} info for this participant.")
                pad.loc[pad['prolific_ID'] == participant, variable] = variable_info.loc[variable_info['prolific_ID'] == participant, variable].values[0]
            
    # concatenate the original dataframe with the new dataframe with rows for the missing participants
    df_updated = pd.concat([df, pad], ignore_index=True)

    #  check that 'prolific_ID' and 'exp_ID' columns in the updated dataframe do not have any missing values
    if df_updated['prolific_ID'].isnull().any():
        raise ValueError("Error: After adding missing participants, there are missing values in the 'prolific_ID' column.")
    if 'exp_ID' in df_updated.columns and df_updated['exp_ID'].isnull().any():
        raise ValueError("Error: After adding missing participants, there are still missing values in the 'exp_ID' column.")

    #  check that there are now the same participant IDs in the updated dataframe and in the set of all participants
    updated_dataset_participants = set(df_updated['prolific_ID'].unique())
    if updated_dataset_participants != set(LearningTask['prolific_ID']):
        print("\nError: After adding missing participants, the participant IDs in the updated dataframe do not match the set of all participants.")
        print("Participant IDs in updated dataframe but not in set of all participants:", updated_dataset_participants - set(LearningTask['prolific_ID']))
        print("Participant IDs in§ set of all participants but not in updated dataframe:", set(LearningTask['prolific_ID']) - updated_dataset_participants)
        raise ValueError("Error: After adding missing participants, the participant IDs in the updated dataframe do not match the set of all participants.")
    
    return df_updated