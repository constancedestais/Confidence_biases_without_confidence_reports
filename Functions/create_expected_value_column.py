def create_expected_value_column(df):
    """
    This function creates a new column in the DataFrame that contains the expected value of the pair on a given trial (works for Learning Task and Symbol Choice task).

    INPUTS:
    df (pd.DataFrame): The DataFrame containing the data.

    OUTPUTS:
    pd.DataFrame: The DataFrame with the new column in the DataFrame that contains the expected value of each row.

    The expected value is calculated as the product of the probability and the possible outcomes for each row.
    """
    import pandas as pd
    
    # Calculate expected value - need to round to deal with floating point precision issues: round(result, 2)
    df['symbol_chosen_expected_value']   = round(df['symbol_chosen_best_outcome']   * df['symbol_chosen_probability_best_outcome']   +  df['symbol_chosen_worst_outcome']   * (1-df['symbol_chosen_probability_best_outcome']), 4)
    df['symbol_unchosen_expected_value'] = round(df['symbol_unchosen_best_outcome'] * df['symbol_unchosen_probability_best_outcome'] +  df['symbol_unchosen_worst_outcome'] * (1-df['symbol_unchosen_probability_best_outcome']), 4)
    df['expected_value'] = round( (df['symbol_chosen_expected_value']+df['symbol_unchosen_expected_value'])/2, 4)
    
    return df