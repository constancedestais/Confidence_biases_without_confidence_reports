from Functions.custom_timeseries_plots import create_timeseries_plot


def plot_pchosehighestEV_over_trials_LT(df_mean_per_participant_per_trial):
    '''
    INPUTS:
    df_mean_per_participant_per_trial: DataFrame containing the mean probability of choosing the highest expected value - averaged per trial per participant per valence condition

    OUTPUTS:
    plt object
    '''
    
    return create_timeseries_plot(
        df=df_mean_per_participant_per_trial,
        id_column='participant_ID',
        trial_column='trial_per_pair',
        value_column='chose_highest_expected_value',
        condition_column=None,
        condition_mapping=None,
        condition_colors=None,
        ylabel='chose highest EV (%)',
        reference_value=50,
        one_indexed=True,
        x_ticks=None, 
        y_ticks=None,
        x_limits=None, 
        y_limits=[0,100], 
        show_legend=False, 
        fig_size=(3, 2),
        font_size= 14,
        line_width= 2,
        line_alpha= 0.4,              
        midline_color= 'black',
        font_name= 'Arial',
    )