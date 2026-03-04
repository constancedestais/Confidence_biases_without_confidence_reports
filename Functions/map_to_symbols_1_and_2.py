def map_to_symbols_1_and_2(df_LearningTask):

    #%% Create columns for symbol_1 and symbol_2 features and outcomes
    
    # Extract symbol IDs directly from the symbol_pair tuple
    df_LearningTask['symbol_1_id'] = df_LearningTask['symbol_pair'].apply(lambda x: x[0])
    df_LearningTask['symbol_2_id'] = df_LearningTask['symbol_pair'].apply(lambda x: x[1])

    # Create new columns for Symbol 1
    attribute_columns = ['best_outcome', 'worst_outcome', 'probability_best_outcome', 'outcome', 'expected_value']

    # Function to determine if symbol_1 is the chosen or unchosen symbol in a row, and to map the corresponding attribute
    def map_symbol_1_attributes(row, attribute):
        if row['symbol_1_id'] == row['symbol_chosen_id']:
            return row[f'symbol_chosen_{attribute}']
        else:
            return row[f'symbol_unchosen_{attribute}']

    # Function to determine if symbol_2 is the chosen or unchosen symbol in a row, and to map the corresponding attribute
    def map_symbol_2_attributes(row, attribute):
        if row['symbol_2_id'] == row['symbol_chosen_id']:
            return row[f'symbol_chosen_{attribute}']
        else:
            return row[f'symbol_unchosen_{attribute}']

    # Create the new columns
    for attr in attribute_columns:
        df_LearningTask[f'symbol_1_{attr}'] = df_LearningTask.apply(lambda row: map_symbol_1_attributes(row, attr), axis=1)
        df_LearningTask[f'symbol_2_{attr}'] = df_LearningTask.apply(lambda row: map_symbol_2_attributes(row, attr), axis=1)


    #%% Create column for chose_symbol_1 
    def symbol_1_same_as_chosen_symbol(row):
        if row['symbol_1_id'] == row['symbol_chosen_id']:
            return 1
        else:
            return 0
    df_LearningTask[f'chose_symbol_1'] = df_LearningTask.apply(lambda row: symbol_1_same_as_chosen_symbol(row), axis=1)

    #%%
    return df_LearningTask