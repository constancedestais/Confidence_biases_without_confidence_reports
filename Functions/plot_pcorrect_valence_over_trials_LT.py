from Functions.custom_timeseries_plots import create_timeseries_plot
from Functions.my_colors import load_my_colors


def plot_pcorrect_valence_over_trials_LT(df_mean_per_participant_per_trial):
    """Your specific time series plotting function using the generic function"""

    c = load_my_colors()

    return create_timeseries_plot(
        df=df_mean_per_participant_per_trial,
        id_column='participant_ID',
        condition_column='is_gain_trial',
        trial_column='trial_per_pair',
        value_column='correct',
        condition_mapping={True: 'Gain', False: 'Loss'},
        condition_colors={'Gain': c["medium_green"], 'Loss': c["medium_red"]}, #condition_colors={'Gain': c["medium_turquoise"], 'Loss': c["medium_purple"]}, 
        reference_value=50,
        # labels and ticks
        ylabel='accuracy (%)',
        y_limits=[25,100],
        y_ticks=[25, 50, 75, 100], 
        x_ticks=[1, 10, 20, df_mean_per_participant_per_trial['trial_per_pair'].nunique()], 
        # aesthetics
        figure_size=(1.5, 1), # width, height, in inches
        font_name= 'Arial',
        font_size= 8,
        line_width= 1.1,
        line_alpha= 0.9,              
        midline_color= 'black',
        one_indexed=True,
        show_legend=False, 

    )



