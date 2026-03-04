from Functions.custom_boxplots import create_single_boxplot
from Functions.raincloud_plot_DEPRECATED import raincloud_plot 
from Functions.from_df_to_datacell import (df_to_array_no_conditions)    
import numpy as np
from Functions.raincloud_unpaired import raincloud_unpaired
from Functions.my_colors import load_my_colors
from matplotlib import font_manager as fm

def plot_chose_gain_pair_CFC(df_mean_per_participant):

    """Your CFC plotting function using the generic single boxplot function"""
    '''
    # Verify that each participant appears exactly once
    validation_counts = df_mean_per_participant.groupby('participant_ID').size()
    try:
        assert (validation_counts == 1).all(), "Error: Some participants have multiple entries"
        print("\nDataset validation passed: One entry per participant")
    except AssertionError as e:
        print(e)
        problem_entries = validation_counts[validation_counts > 1].reset_index()
        print("Problematic entries (duplicates):")
        print(problem_entries)
        raise ValueError("Dataset validation failed - see above for details")
    
    # Settings specific to this plot
    cfc_settings = {
        'figsize': (3, 4),
        'fontsize': 14,
        'boxplot_linewidth': 1.5,
        'boxplot_width': 0.5,
        'alpha': 0.5,
        'point_size': 5,
        'point_color': 'black',
        'boxplot_color': 'forestgreen', #'lightgray',
        'midline_color': 'gray',
        'midline_linestyle': '--',
        'midline_alpha': 0.7,
        'ymin': 0,
        'ymax': 100,
        'star_fontsize': 20
    }
    
    return create_single_boxplot(
        df=df_mean_per_participant,
        value_col='chose_highest_expected_value',
        test_value=50,  # Test against 50% chance level
        multiply_by=100,  # Convert proportions to percentages
        ylabel='Choose gain pair (%)',
        reference_value=50,  # Add reference line at chance
        settings=cfc_settings
    )

    '''
    c = load_my_colors()

    # Verify that each participant appears exactly once
    validation_counts = df_mean_per_participant.groupby('participant_ID').size()
    try:
        assert (validation_counts == 1).all(), "Error: Some participants have multiple entries"
        print("\nDataset validation passed: One entry per participant")
    except AssertionError as e:
        print(e)
        problem_entries = validation_counts[validation_counts > 1].reset_index()
        print("Problematic entries (duplicates):")
        print(problem_entries)
        raise ValueError("Dataset validation failed - see above for details")
    
    # check that variable is not between -1 and 1
    assert df_mean_per_participant["chose_highest_expected_value"].max() > 1.5, "Error: chose_highest_expected_value values look like they are still coded 0-1"

    # check that Arial font is available
    Arial_exists = fm.findfont("Arial", fallback_to_default=False)
    assert Arial_exists, "Arial font is not available. Please install Arial font to proceed."

    # transform df to arrays
    data_list = df_to_array_no_conditions(df_mean_per_participant, 
                                                value_column="chose_highest_expected_value", 
                                                )    

    fig, ax = raincloud_unpaired(data_list,
                                my_colors = (c["dark_green"],), #my_colors=[(0.2, 0.4, 0.8)],
                                # axis labels & ticks
                                y_limits=(0, None),
                                x_tick_labels=(""),
                                y_ticks=[0, 25, 50, 75, 100],
                                #y_ticks=[0, 0.25, 0.50, 0.75, 1.00],
                                #y_tick_labels=["0", "25", "50", "75", "100"],
                                reference_value=50.50,
                                # figure + fonts
                                figure_size=(0.5, 1), # width, height of axis, in inches
                                font_size=8.0,
                                font_name="Arial",
                                # geometry / layout
                                cloud_width = 0.008,   # half-width of each half-violin
                                x_margin = 0.004,
                                # dots
                                dot_size=15.0,
                                dot_alpha=0.2,
                                dot_distance=0.002,
                                # mean / error bars
                                line_width=0.85,
                                sem_cap_width=4.0,
                                # significance stars
                                show_significance_stars=True,
                                star_fontsize=12.0,
                                ns_fontsize=6.0,
                            )

    
    return  fig, ax

