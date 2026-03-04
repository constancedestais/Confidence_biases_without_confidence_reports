from Functions.custom_boxplots import create_paired_boxplot
from Functions.raincloud_plot_DEPRECATED import raincloud_plot 
from Functions.from_df_to_datacell import (df_to_arrays_paired_conditions)    
import numpy as np
from Functions.raincloud_paired_two_conditions import raincloud_paired_two_conditions
from Functions.my_colors import load_my_colors

def plot_pcorrect_valence_SC(df_mean_per_participant):
    
    '''
    # Settings specific to this plot
    my_settings = {
        'figsize': (5, 4),
        'fontsize': 14,
        'boxplot_linewidth': 1.5,
        'boxplot_width': 0.5,
        'jitter': False,
        'dodge': False,
        'marker': 'o',
        'alpha': 0.35,
        'point_size': 5,
        'point_color': 'black',
        'connect_linestyle': '-',
        'connect_linewidth': 0.5,
        'connect_color': 'grey',
        'connect_alpha': 0.35,
        'midline_color': 'grey',
        'midline_linestyle': '--',
        'midline_alpha': 0.7,
        'ymin': 0,
        'ymax': 100,
        'star_fontsize': 20,
        'star_fontsize_ns': 16
    }
    
    return create_paired_boxplot(
        df=df_mean_per_participant,
        id_col='participant_ID',
        condition_col='is_gain_trial',
        value_col='correct',
        condition_mapping={True: 'Gain', False: 'Loss'},
        condition_colors={"Gain": "forestgreen", "Loss": "indianred"},
        multiply_by=100,
        ylabel='accuracy (%)',
        reference_value=50,
        settings=my_settings
    )
    '''
    c = load_my_colors()

    # check that variable is not between -1 and 1
    assert df_mean_per_participant["correct"].max() > 1.5, "Error: correct values look like they are still coded 0-1"

    # transform df to arrays
    data_list, order = df_to_arrays_paired_conditions(df_mean_per_participant, 
                                                        condition_column="is_gain_trial", 
                                                        value_column="correct", 
                                                        subject_column="participant_ID",
                                                        condition_mapping={True: 'Positive', False: 'Negative'},
                                                        condition_order=[True,False]  # optional but recommended for stable order
                                                        )    

    # call plotting function
    fig, ax = raincloud_paired_two_conditions(
        data_list,
        my_colors=(c["dark_green"], c["dark_red"]), #my_colors=(c["medium_turquoise"], c["medium_purple"]), #my_colors=("forestgreen", "indianred"),
        reference_value=50.0,
        # axis labels & ticks
        y_limits=(0, None),
        x_tick_labels=("Positive", "Negative"),
        y_ticks=[0, 25, 50, 75, 100],
        # figure + fonts
        figure_size=(0.75, 1), # width, height of axis, in inches
        font_size=8.0,
        font_name="Arial",
        # geometry / layout
        cloud_width = 0.010,   # half-width of each half-violin
        pair_gap=0.018,        # distance between the two clouds
        # dots
        dot_size=15.0,
        dot_alpha=0.2,
        dot_distance=0.003,    # distance from violin axis to dot column
        # mean / error bars
        line_width=0.85,
        sem_cap_width=4.0,
        # connecting lines
        connect_linewidth=0.3,
        connect_alpha=0.8,
        highlight_connected_lines_in_dominant_direction=False,
        # significance stars
        show_significance_stars=True,
        star_fontsize=12.0,
        ns_fontsize=6.0,
    )

    return fig, ax