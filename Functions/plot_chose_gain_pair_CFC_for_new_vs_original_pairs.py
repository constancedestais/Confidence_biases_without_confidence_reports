import pandas as pd
import numpy as np
from Functions.raincloud_paired_two_conditions import raincloud_paired_two_conditions
from Functions.my_colors import load_my_colors

def plot_chose_gain_pair_CFC_for_new_vs_original_pairs(CFC_mean_per_participant_by_includes_new_pair):

    # check that variable is not between 0 and 1 (and thus is probably scaled to percentage)
    assert CFC_mean_per_participant_by_includes_new_pair[0].max() > 1.5, f"Error: chose_highest_expected_value values look like they are still coded 0-1" 
    assert CFC_mean_per_participant_by_includes_new_pair[1].max() > 1.5, f"Error: chose_highest_expected_value values look like they are still coded 0-1"

    c = load_my_colors() 
    
   
    # make plot: CFC_chose_highest_expected_value for new vs original pairs

    fig, ax = raincloud_paired_two_conditions(
            CFC_mean_per_participant_by_includes_new_pair,
            my_colors=(c["dark_green"], c["dark_green"]), #my_colors=(c["medium_turquoise"], c["medium_purple"]), #my_colors=("forestgreen", "indianred"),
            reference_value=50.0,
            # axis labels & ticks
            y_limits=(0, None),
            x_tick_labels=("new", "old"),
            y_ticks=[0, 25, 50, 75, 100],
            # figure + fonts
            figure_size=(0.5, 1), # width, height of axis, in inches
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
    fig.show()

    return fig, ax