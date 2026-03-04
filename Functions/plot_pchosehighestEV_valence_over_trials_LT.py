from Functions.custom_timeseries_plots import create_timeseries_plot

def plot_pchosehighestEV_valence_over_trials_LT(df_mean_per_participant_per_trial):
    '''
    INPUTS:
    df_mean_per_participant_per_trial: DataFrame containing the mean probability of choosing the highest expected value - averaged per trial per participant per valence condition

    OUTPUTS:
    plt object
    '''

    '''
    # Settings specific to this plot
    timeseries_settings = {
        'figsize': (6, 4),
        'fontsize': 16,
        'linewidth': 2,
        'line_alpha': 0.4,
        'marker_alpha': 0.6,
        'markersize': 8,
        'ymin': 0,
        'ymax': 100
    }
    
    return create_timeseries_plot(
        df=df_mean_per_participant_per_trial,
        id_col='participant_ID',
        condition_col='is_gain_trial',
        trial_col='trial_per_pair',
        value_col='chose_highest_expected_value',
        condition_mapping={True: 'Gain', False: 'Loss'},
        condition_colors={'Gain': 'green', 'Loss': 'red'},
        condition_markers={'Gain': 'o', 'Loss': 's'},
        ylabel='chose highest EV (%)',
        reference_value=50,
        one_indexed=True,
        settings=timeseries_settings
    )
    '''

    
    return create_timeseries_plot(
        df=df_mean_per_participant_per_trial,
        id_column='participant_ID',
        trial_column='trial_per_pair',
        value_column='chose_highest_expected_value',
        condition_column='is_gain_trial',
        condition_mapping={True: 'Gain', False: 'Loss'},
        condition_colors={'Gain': 'green', 'Loss': 'red'},
        ylabel='chose highest EV (%)',
        reference_value=50,
        one_indexed=True,
        y_limits=[0,100], 
        show_legend=False, 
        fig_size=(3, 2),
        font_size= 14,
        line_width= 2,
        line_alpha= 0.4,              
        midline_color= 'black',
        font_name= 'Arial',
    )