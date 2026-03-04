from Functions.custom_timeseries_plots import create_timeseries_plot

def plot_pcorrect_over_trials_LT(df_mean_per_participant_per_trial):
    '''
    INPUTS:
    df_mean_per_participant_per_trial: DataFrame containing the mean probability of choosing the CORRECT value - averaged per trial per participant per valence condition

    OUTPUTS:
    plt object
    '''

    '''
    # ----- UNDERSTANDING WHY STD IS DIFFERENT WHEN USING GROUPBY/AGG VS. NUMPY -----
    # show dataset for one trial
    a = df_mean_per_participant_per_trial[df_mean_per_participant_per_trial['trial_per_pair'] == 15]
    # compute std manually
    correct_column = a['correct']
    N = len(a)
    mean = np.mean(correct_column)
    d2 = abs(correct_column - mean)**2  # abs is for complex `a`
    ddof = 1
    var = d2.sum() / (N - ddof)  # note use of `ddof`
    std1 = var**0.5
    # compute std using numpy
    std2 = np.std(a['correct'])
    # compute std using groupby/agg
    b = a.groupby(['trial_per_pair',])['correct'].agg(['std']).reset_index()
    # CONCLUSION: numpy's std assumed ddof=0, while groupby/agg assumed ddof=1
    '''

    '''
    # ------ plotting using box plots -----
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
        id_column='participant_ID',
        condition_column=None,
        trial_column='trial_per_pair',
        value_column='correct',
        ylabel='accuracy (%)',
        reference_value=50,
        one_indexed=True,
        settings=timeseries_settings
    )

    '''

    return create_timeseries_plot(
        df=df_mean_per_participant_per_trial,
        id_column='participant_ID',
        condition_column=None,
        trial_column='trial_per_pair',
        value_column='correct',
        condition_mapping=None,
        condition_colors=None,
        ylabel='accuracy (%)',
        reference_value=50,
        one_indexed=True,
        y_limits=[0,100], 
        show_legend=False, 
        font_size= 14,
        line_width= 2,
        line_alpha= 0.4,              
        midline_color= 'black',
        font_name= 'Arial',
        fig_size=(3, 2),
    )