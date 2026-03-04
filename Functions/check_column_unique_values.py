

def check_column_unique_values_LearningTask(df, column_name):
    """
    Check the unique values in a column and print the type of each value.
    """
    
    column_data = df[column_name]
    # Get the unique values in the column
    unique_values = column_data.unique()
    """
    # Print the unique values and their types
    print(f"Unique values in {column_name}:")
    for value in unique_values:
        print(f"Value: {value}, Type: {type(value)}")
    
    # Print the number of unique values
    print(f"Number of unique values in {column_name}: {len(unique_values)}\n")
    """ 
    if column_name == 'prolific_ID':
        # there should be as many unique values in the prolific_ID column as th
        # assert len(unique_values) == 80, f"\nError: {len(unique_values)} unique values in {column_name} instead of 80"
        # check that all values are strings
        assert all(isinstance(value, str) for value in unique_values), f"\nError: Not all values in {column_name} are strings"
    elif column_name == 'click_desired':
        # check that all unique values are 0 or 1
        assert all(value in [0, 1] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"
    elif column_name == 'n_trials_per_session':
        # check that all unique values are 0 or 1
        assert all(value in [24,30,100] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1" # TO CHANGE BACK
    elif column_name == 'n_sessions':
        # check that all unique values are 0 or 1
        assert all(value in [2] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"
    elif column_name == 'session':
        # check that all unique values are 0 or 1
        assert all(value in [0, 1] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"
    elif column_name == 'trial':
        assert len(unique_values) == 100, f"\nError: {len(unique_values)} unique values in {column_name} instead of 100"
    elif column_name == 'trial_per_cycle':
        assert len(unique_values) == 4, f"\nError: {len(unique_values)} unique values in {column_name} instead of 4"
    elif column_name == 'is_gain_trial':
        # check that all unique values are 0 or 1
        assert all(value in [0, 1] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"
    elif (column_name == 'key_top') | (column_name == 'key_bottom') | (column_name == 'response_key'):
        assert all(value in ['s', 'k'] for value in unique_values), f"\nError: Not all values in {column_name} are s or k"
    elif column_name == 'responded_bottom':
        # check that all unique values are 0 or 1
        assert all(value in [0, 1] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"
    elif (column_name == 'symbol_chosen_id') | (column_name == 'symbol_unchosen_id') :
        assert all(value in [2,4,1,3,0,7,5,6] for value in unique_values), f"\nError: Not all values in {column_name} are between 0 and 7"
        # check that there are 8 unique values
        assert len(unique_values) == 8, f"\nError: {len(unique_values)} unique values in {column_name} instead of 8"
    elif (column_name == 'symbol_chosen_imageID') | (column_name == 'symbol_unchosen_imageID'):
        assert all(value in [2,4,1,3,0,7,5,6] for value in unique_values), f"\nError: Not all values in {column_name} are between 0 and 7"
    elif (column_name == 'symbol_chosen_best_outcome') | (column_name == 'symbol_unchosen_best_outcome'):
        assert all(value in [1,-0.1] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"
    elif (column_name == 'symbol_chosen_worst_outcome') | (column_name == 'symbol_unchosen_worst_outcome'):
        assert all(value in [0.1,-1] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"
    elif (column_name == 'symbol_chosen_outcome') |  (column_name == 'symbol_unchosen_outcome'):
        assert all(value in [0.1,1.,-0.1,-1.] for value in unique_values), f"\nError: Not all values in {column_name} are 0.1,1,-0,1,-1"
    elif (column_name == 'symbol_chosen_probability_best_outcome') | (column_name == 'symbol_unchosen_probability_best_outcome'):
        assert all(value in [0.75,0.25,0.70,0.30,0.80,0.20] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"

def check_column_unique_values_PairChoice(df, column_name):
    """
    Check the unique values in a column and print the type of each value.
    """
    column_data = df[column_name]
    # Get the unique values in the column
    unique_values = column_data.unique()

    if column_name == 'identify_best':
        # check that all unique values are 0 or 1
        assert all(value in [0, 1] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"
    elif column_name == 'n_sessions':
        # check that all unique values are 0 or 1
        assert all(value in [1] for value in unique_values), f"\nError: Not all values in {column_name} are 1"
    elif column_name == 'n_trials_per_session':
        # check that all unique values are 0 or 1
        assert all(value in [128] for value in unique_values), f"\nError: Not all values in {column_name} are 128"
    elif column_name == 'trial':
        assert len(unique_values) == 128, f"\nError: {len(unique_values)} unique values in {column_name} instead of 128"
    elif column_name == 'trial_per_cycle':
        assert len(unique_values) == 32, f"\nError: {len(unique_values)} unique values in {column_name} instead of 32"
    elif column_name == 'session':
        assert all(value in [0] for value in unique_values), f"\nError: Not all values in {column_name} are 0"
    elif column_name == 'responded_right_side':
        assert all(value in [0,1] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"
    elif column_name == 'response_key':
        assert all(value in ['k','s'] for value in unique_values), f"\nError: Not all values in {column_name} are s or k"
    elif (column_name == 'pair_chosen_is_new_pair') | (column_name == 'pair_unchosen_is_new_pair'):
        assert all(value in [0,1] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"
    elif (column_name == 'pair_chosen_symbol_top_id') | (column_name == 'pair_unchosen_symbol_top_id') :
        assert all(value in [0,1,2,3,4,5,6,7] for value in unique_values), f"\nError: Not all values in {column_name} are 0,1,2,3,4,5,6,7"
        # check that there are 8 unique values
        assert len(unique_values) == 8, f"\nError: {len(unique_values)} unique values in {column_name} instead of 8"
    elif (column_name == 'pair_chosen_symbol_top_best_outcome') | (column_name == 'pair_chosen_symbol_bottom_best_outcome') | (column_name == 'pair_unchosen_symbol_top_best_outcome') | (column_name == 'pair_unchosen_symbol_bottom_best_outcome') :
        assert all(value in [-0.1,1] for value in unique_values), f"\nError: Not all values in {column_name} are -0.1 or 1"
    elif (column_name == 'pair_chosen_symbol_top_worst_outcome') | (column_name == 'pair_chosen_symbol_bottom_worst_outcome') | (column_name == 'pair_unchosen_symbol_top_worst_outcome') | (column_name == 'pair_unchosen_symbol_bottom_worst_outcome') :
        assert all(value in [-1,0.1] for value in unique_values), f"\nError: Not all values in {column_name} are -1 or 0.1"
    elif (column_name == 'pair_chosen_symbol_top_probability_best_outcome') | (column_name == 'pair_chosen_symbol_bottom_probability_best_outcome') | (column_name == 'pair_unchosen_symbol_top_probability_best_outcome') | (column_name == 'pair_unchosen_symbol_bottom_probability_best_outcome') :
        assert all(value in [0.75,0.25,0.70,0.30,0.80,0.20] for value in unique_values), f"\nError: Not all values in {column_name} are 0.75,0.25,0.70,0.30,0.80,0.20"
    elif (column_name == 'pair_chosen_symbol_top_is_gain') | (column_name == 'pair_chosen_symbol_bottom_is_gain') | (column_name == 'pair_unchosen_symbol_top_is_gain') | (column_name == 'pair_unchosen_symbol_bottom_is_gain') :
        assert all(value in [0,1] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"
    elif (column_name == 'pair_chosen_expected_value') | (column_name == 'pair_unchosen_expected_value') :
        assert all(value in [-0.55,0.55,0.225,-0.225] for value in unique_values), f"\nError: Not all values in {column_name} are -0.55,0.55,0.225,-0.225"
    elif (column_name == 'chose_highest_expected_value') | (column_name == 'pair_unchosen_expected_value') :
        assert all(value in [0,1] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"

def check_column_unique_values_SymbolChoice(df, column_name):
    """
    Check the unique values in a column and print the type of each value.
    """
    column_data = df[column_name]
    # Get the unique values in the column
    unique_values = column_data.unique()

    if column_name == 'identify_best':
        # check that all unique values are 0 or 1
        assert all(value in [0, 1] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"
    elif column_name == 'responded_right_side':
        assert all(value in [0,1] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"
    elif column_name == 'n_sessions':
        assert all(value in [1] for value in unique_values), f"\nError: Not all values in {column_name} are 1"
    elif column_name == 'trial':
        assert len(unique_values) == 64, f"\nError: {len(unique_values)} unique values in {column_name} instead of 64"
    elif column_name == 'trial_per_cycle':
        assert len(unique_values) == 16, f"\nError: {len(unique_values)} unique values in {column_name} instead of 64"
    elif column_name == 'session':
        assert all(value in [0] for value in unique_values), f"\nError: Not all values in {column_name} are 1"
    elif column_name == 'is_new_pair':
        assert all(value in [0,1] for value in unique_values), f"\nError: Not all values in {column_name} are 1"
        # For each participant, 1/4th of the values should be 0, and 3/4th should be 1
        # Count occurrences of each is_new_pair value per prolific_ID
        pair_counts = df.groupby('prolific_ID')['is_new_pair'].value_counts().unstack(fill_value=0)
        # Display participants who do NOT meet the expected counts
        invalid_participants = pair_counts[(pair_counts[0] != 16) | (pair_counts[1] != 48)]
        # Print the results
        if not invalid_participants.empty:
            print("\nParticipants with incorrect is_new_pair counts:")
            print(invalid_participants)
        #else:
            #print("All participants have correct is_new_pair counts (16 zeros and 48 ones).")
    elif column_name == 'mixed_symbol_valence':
        assert all(value in [0,1] for value in unique_values), f"\nError: Not all values in {column_name} are 1"
        # For each participant, half of the values should be 0, and half should be 1
        # Count occurrences of each mixed_symbol_valence value per prolific_ID
        pair_counts = df.groupby('prolific_ID')['mixed_symbol_valence'].value_counts().unstack(fill_value=0)
        # Display participants who do NOT meet the expected counts
        invalid_participants = pair_counts[(pair_counts[0] != 32) | (pair_counts[1] != 32)]
        # Print the results
        if not invalid_participants.empty:
            print("\nParticipants with incorrect mixed_symbol_valence counts:")
            print(invalid_participants)
        #else:
            #print("All participants have correct mixed_symbol_valence counts (32 zeros and 32 ones).")
    elif (column_name == 'symbol_chosen_id') | (column_name == 'symbol_chosen_bottom_id') :
        assert all(value in [0,1,2,3,4,5,6,7] for value in unique_values), f"\nError: Not all values in {column_name} are between 0 and 7"
    elif (column_name == 'symbol_chosen_probability_best_outcome') | (column_name == 'symbol_unchosen_probability_best_outcome') :
        assert all(value in [0.25,0.75,0.70,0.30,0.80,0.20] for value in unique_values), f"\nError: Not all values in {column_name} are 0.75,0.25,0.70,0.30,0.80,0.20" # TO CHANGE BACK
    elif (column_name == 'symbol_chosen_best_outcome') | (column_name == 'symbol_unchosen_best_outcome') :
        assert all(value in [-0.1,1] for value in unique_values), f"\nError: Not all values in {column_name} are -0.1 or 1"
    elif (column_name == 'symbol_chosen_worst_outcome') | (column_name == 'symbol_unchosen_worst_outcome') :
        assert all(value in [-1,0.1] for value in unique_values), f"\nError: Not all values in {column_name} are -1 or 0.1"
    elif (column_name == 'response_key') :
        assert all(value in ['s','k'] for value in unique_values), f"\nError: Not all values in {column_name} are s or k"
    elif (column_name == 'responded_right_side') :
        assert all(value in [0,1] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"
    elif (column_name == 'chose_highest_expected_value') :
        assert all(value in [0,1] for value in unique_values), f"\nError: Not all values in {column_name} are 0 or 1"
