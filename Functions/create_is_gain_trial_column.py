def create_is_gain_trial_column(df):
    """
    Create a new column in the DataFrame that indicates whether the trial is a gain trial (expected value >= 0) or a loss trial (expected value < 0).

    INPUTS:
    df (pd.DataFrame): The DataFrame containing the data, and it must contain a column named 'expected_value'.

    OUTPUTS:
    pd.DataFrame: The DataFrame with the new 'is_gain_trial' column added.
    """

    # check if 'expected_value' column exists
    if 'expected_value' not in df.columns:
        raise ValueError("The DataFrame must contain a column named 'expected_value'.")
    
    # Create the 'is_gain_trial' column based on the condition
    df['is_gain_trial'] = df['expected_value'] >= 0
    
    return df