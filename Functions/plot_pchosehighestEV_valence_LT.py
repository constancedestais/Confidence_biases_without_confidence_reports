from Functions.custom_boxplots import create_paired_boxplot
from Functions.raincloud_plot_DEPRECATED import raincloud_plot 
from Functions.from_df_to_datacell import (df_to_arrays_paired_conditions)    
import numpy as np

def plot_pchosehighestEV_valence_LT(df_mean_per_participant_per_valence):
    '''
    main variable should be scaled to percentage (0 to 100 or -100 to 100) before being passed to this function, so that the reference line at 50.0 makes sense. 
    
    '''
        
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
        'alpha': 0.5,
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
        df=df_mean_per_participant_per_valence,
        id_col='participant_ID',
        condition_col='is_gain_trial',
        value_col='chose_highest_expected_value',
        condition_mapping={True: 'Gain', False: 'Loss'},
        condition_colors={"Gain": "forestgreen", "Loss": "indianred"},
        multiply_by=100,
        ylabel='Chose highest EV (%)',
        reference_value=50,
        settings=my_settings
    )
    '''
    
    # check that variable is not between 0 and 1 (and thus is probably scaled to percentage)
    assert df_mean_per_participant_per_valence['chose_highest_expected_value'].max() > 1.5, f"Error: chose_highest_expected_value values look like they are still coded 0-1"


    # transform df to arrays
    data_list, order = df_to_arrays_paired_conditions(df_mean_per_participant_per_valence, 
                                                    condition_column="is_gain_trial", 
                                                    value_column="chose_highest_expected_value", 
                                                    subject_column="participant_ID",
                                                    condition_mapping={True: 'Gain', False: 'Loss'},
                                                    condition_order=[True,False]  # optional but recommended for stable order
                                                    )    

    # call plotting function
    ax,fig = raincloud_plot(plot_type=2,
                            DataCell=data_list,
                            xRefInput= order,#np.nan,
                            my_colors=["forestgreen", "indianred"],
                            Yinf= 20,
                            Ysup= 100,
                            font_size=14,
                            Title="",
                            LabelX="",
                            LabelY='accuracy (%)',
                            x_tick_labels=order,
                            dot_size=40,
                            sem_bar_width=12,
                            line_width=2,
                            highlight_connected_lines_in_dominant_direction=True,
                            base_distance_between_datapoints_and_violin=0.2,
                            connect_linewidth=1,   # linewidth for connecting lines
                            connect_alpha=0.35,       # alpha for connecting lines (overrides defaults)
                            point_color='grey',         # color for data points; single color or list length Nbar
                            point_alpha=0.5,         # alpha for data points
                            show_significance_stars=True,
                            star_fontsize=15,       # fontsize when significant (*, **, ***)
                            ns_fontsize=12,         # fontsize for 'n.s.' (and 'n.a')
                            reference_value=50,
                            figure_size=(3,2)
                        )

    return ax, fig
