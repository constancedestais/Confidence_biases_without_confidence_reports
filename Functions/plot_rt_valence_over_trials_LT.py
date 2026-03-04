import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from Functions.custom_timeseries_plots import create_timeseries_plot

def plot_rt_valence_over_trials_LT(df_median_per_participant_per_trial_per_valence):
    '''
    INPUTS:
    df_median_per_participant_per_trial_per_valence: DataFrame containing the mean probability of choosing the highest expected value - averaged per trial per participant per valence condition

    OUTPUTS:
    plt object
    '''

    '''
    # Settings specific to this plot
    timeseries_settings = {
        'averaging': 'median', # 'mean' or 'median'
        'figsize': (6, 4),
        'fontsize': 16,
        'linewidth': 2,
        'line_alpha': 0.4,
        'marker_alpha': 0.6,
        'markersize': 8,
    }
    
    return create_timeseries_plot(
        df=df_median_per_participant_per_trial_per_valence,
        id_column='participant_ID',
        condition_column='is_gain_trial',
        trial_column='trial_per_pair',
        value_column='rt',
        condition_mapping={True: 'Gain', False: 'Loss'},
        condition_columnors={'Gain': 'green', 'Loss': 'red'},
        condition_markers={'Gain': 'o', 'Loss': 's'},
        ylabel='RT (ms)',
        reference_value=None,
        one_indexed=True,
        settings=timeseries_settings
    )


    '''

    return create_timeseries_plot(
        df=df_median_per_participant_per_trial_per_valence,
        id_column='participant_ID',
        condition_column='is_gain_trial',
        trial_column='trial_per_pair',
        value_column='rt',
        condition_mapping={True: 'Gain', False: 'Loss'},
        condition_colors={'Gain': 'green', 'Loss': 'red'},
        ylabel='accuracy (%)',
        reference_value=None,
        one_indexed=True,
        y_limits=None, 
        show_legend=False, 
        averaging='mean',
        fig_size=(3, 2),
        font_size= 14,
        line_width= 2,
        line_alpha= 0.4,              
        midline_color= 'black',
        font_name= 'Arial',
        )