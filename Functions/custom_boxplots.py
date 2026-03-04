import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math

from Functions.significance_stars import significance_stars


def create_paired_boxplot(df, id_col, condition_col, value_col, 
                         condition_mapping=None, condition_colors=None,
                         multiply_by=1, ylabel="Value", title="",
                         reference_value=None, midline_label="",
                         settings=None):
    """
    Creates a paired boxplot with connecting lines between conditions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    id_col : str
        Column name for participant/subject IDs
    condition_col : str  
        Column name for the condition (0/1 or categorical)
    value_col : str
        Column name for the values to plot
    condition_mapping : dict, optional
        Mapping for condition values (e.g., {1: 'Gain', 0: 'Loss'})
    condition_colors : dict, optional
        Colors for each condition (e.g., {"Gain": "lightgreen", "Loss": "lightcoral"})
    multiply_by : float, default 1
        Multiply values by this (e.g., 100 to convert proportions to percentages)
    ylabel : str, default "Value"
        Y-axis label
    title : str, default ""
        Plot title
    reference_value : float, optional
        Add horizontal reference line at this value
    midline_label : str, default ""
        Label for the midline
    settings : dict, optional
        Dictionary of plot settings (see default_settings for options)
    
    Returns:
    --------
    matplotlib.pyplot object
    """
    
    # Default settings
    default_settings = {
        'figsize': (4, 4),
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
        'connect_alpha': 0.5,
        'midline_color': 'grey',
        'midline_linestyle': '--',
        'midline_alpha': 0.7,
        'ymin': None,
        'ymax': None,
        'star_fontsize': 20,
        'star_fontsize_ns': 16
    }
    
    # Update settings with user provided values
    if settings is None:
        settings = {}
    my_settings = {**default_settings, **settings}
    
    # Prepare data
    df_plot = df[[id_col, condition_col, value_col]].copy()
    
    # Apply condition mapping if provided
    if condition_mapping != None:
        df_plot['condition'] = df_plot[condition_col].map(condition_mapping)
    else:
        df_plot['condition'] = df_plot[condition_col]
    
    # Multiply values if needed
    df_plot[value_col] *= multiply_by
    
    # Set up the figure
    plt.figure(figsize=my_settings['figsize'])
    
    # Create the boxplots
    ax = sns.boxplot(x='condition', y=value_col, data=df_plot,
                    palette=condition_colors,
                    width=my_settings['boxplot_width'],
                    showfliers=False,
                    linewidth=my_settings['boxplot_linewidth'],
                    medianprops={'linewidth': my_settings['boxplot_linewidth']*2})
    
    # Add individual data points
    sns.stripplot(x='condition', y=value_col, data=df_plot,
                  jitter=my_settings['jitter'],
                  dodge=my_settings['dodge'],
                  marker=my_settings['marker'],
                  alpha=my_settings['alpha'],
                  color=my_settings['point_color'],
                  size=my_settings['point_size'])
    
    # Connect data points for same participant across conditions
    pivot_data = df_plot.pivot(index=id_col, columns='condition', values=value_col)
    # CONSTANCE, NEED TO invert order of Gain/Loss columns to match previous plotting order where Loss is plotted first
    # Get unique conditions in their original order of appearance
    condition_order = df_plot[condition_col].map( condition_mapping if condition_mapping else dict(zip(df_plot[condition_col], df_plot[condition_col])) ).drop_duplicates()
    # Reorder columns to match original order
    pivot_data = pivot_data[condition_order]
    conditions = list(pivot_data.columns)

    
    if len(conditions) == 2:
        for participant in pivot_data.index:
            if not pd.isna(pivot_data.loc[participant, conditions[0]]) and \
               not pd.isna(pivot_data.loc[participant, conditions[1]]):
                plt.plot([0, 1], 
                        [pivot_data.loc[participant, conditions[0]], 
                         pivot_data.loc[participant, conditions[1]]],
                        color=my_settings['connect_color'],
                        linestyle=my_settings['connect_linestyle'],
                        linewidth=my_settings['connect_linewidth'],
                        alpha=my_settings['connect_alpha'])
                a=1
    
    # Labels and formatting
    plt.xlabel('', fontsize=my_settings['fontsize'])
    plt.ylabel(ylabel, fontsize=my_settings['fontsize'])
    if title:
        plt.title(title, fontsize=my_settings['fontsize'])
    
    # Add midline if specified
    if reference_value is not None:
        plt.axhline(y=reference_value, 
                   color=my_settings['midline_color'],
                   linestyle=my_settings['midline_linestyle'],
                   alpha=my_settings['midline_alpha'])
    
    # Style the plot
    plt.grid(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    

    #%% Statistical testing and significance stars
    if len(conditions) == 2:
        data1 = pivot_data[conditions[0]].dropna()
        data2 = pivot_data[conditions[1]].dropna()
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(data1, data2)
        
        # Determine significance stars and their fontsize
        sig_stars = significance_stars(p_value)
        if sig_stars in ("*", "**", "***"):
            star_fontsize = my_settings['star_fontsize']  
        else:
            star_fontsize = my_settings['star_fontsize_ns']
            
        # Calculate positions for significance indicators
        data_max = df_plot[value_col].max()
        if my_settings['ymax'] is not None:
            y_max = my_settings['ymax']
            y_min = my_settings['ymin']
        else:
            y_max = data_max * 1.1
            y_min = data_max * 1.1
            
        bar_height = y_max + (abs(y_max - y_min)) * 0.02
        star_height = bar_height + (abs(y_max - y_min)) * 0.035
        
        # Draw significance bar
        plt.plot([0, 1], [bar_height, bar_height], 'k-', linewidth=2)
        
        # Add significance stars
        plt.text(0.5, star_height, sig_stars, ha='center', va='center',
                fontsize=star_fontsize, fontweight='bold')
        
        # Set y-limits
        if my_settings['ymin'] is not None and my_settings['ymax'] is not None:
            plt.ylim(my_settings['ymin'], star_height + (abs(y_max - y_min)) * 0.06)
        
        # Print results in format: Paired t-test (Condition1 vs Condition2): t(df): t_stat; p-value: p_value; Significance: sig_stars
        print(f"\nPaired t-test ({conditions[0]} vs {conditions[1]}): t({len(data1)-1}) = {t_stat:.4f} p-value: {p_value:.4f} Significance: {sig_stars}\n")
    
    plt.tight_layout()
    
    # Tick formatting
    plt.tick_params(axis='y', which='major', length=6, 
                    width=1.5, left=True, direction='in')
    plt.tick_params(axis='both', labelsize=my_settings['fontsize'])
    
    return plt









def create_single_boxplot(df, value_col, test_value=None,
                         multiply_by=1, ylabel="Value", title="",
                         reference_value=None, midline_label="",
                         settings=None):
    """
    Creates a single boxplot with significance testing against a reference value.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    value_col : str
        Column name for the values to plot
    test_value : float, optional
        Reference value to test against (e.g., 50 for chance level)
    multiply_by : float, default 1
        Multiply values by this (e.g., 100 to convert proportions to percentages)
    ylabel : str, default "Value"
        Y-axis label
    title : str, default ""
        Plot title
    reference_value : float, optional
        Add horizontal reference line at this value
    midline_label : str, default ""
        Label for the midline
    settings : dict, optional
        Dictionary of plot settings (see default_settings for options)
    
    Returns:
    --------
    matplotlib.pyplot object
    """
    
    # Default settings
    default_settings = {
        'figsize': (4, 4),
        'fontsize': 14,
        'boxplot_linewidth': 1.5,
        'boxplot_width': 0.5,
        'jitter': True,
        'dodge': True,
        'marker': 'o',
        'alpha': 0.5,
        'point_size': 5,
        'point_color': 'black',
        'boxplot_color': 'lightgray',
        'midline_color': 'grey',
        'midline_linestyle': '--',
        'midline_alpha': 0.7,
        'ymin': None,
        'ymax': None,
        'star_fontsize': 20,
        'star_fontsize_ns': 16
    }
    
    # Update settings with user provided values
    if settings is None:
        settings = {}
    my_settings = {**default_settings, **settings}
    
    # Prepare data
    df_plot = df.copy()
    df_plot[value_col] *= multiply_by
    
    # Set up the figure
    plt.figure(figsize=my_settings['figsize'])
    
    # Create the boxplot
    ax = sns.boxplot(y=df_plot[value_col],
                    color=my_settings['boxplot_color'],
                    width=my_settings['boxplot_width'],
                    showfliers=False,
                    linewidth=my_settings['boxplot_linewidth'],
                    medianprops={'linewidth': my_settings['boxplot_linewidth']*2})
    
    # Add individual data points
    sns.swarmplot(y=df_plot[value_col],
                  color=my_settings['point_color'],
                  alpha=my_settings['alpha'],
                  size=my_settings['point_size'])
    
    # Labels and formatting
    plt.xlabel('', fontsize=my_settings['fontsize'])
    plt.ylabel(ylabel, fontsize=my_settings['fontsize'])
    if title:
        plt.title(title, fontsize=my_settings['fontsize'])
    
    # Remove x-axis ticks since it's a single boxplot
    plt.xticks([])
    
    # Add midline if specified
    if reference_value is not None:
        plt.axhline(y=reference_value, 
                   color=my_settings['midline_color'],
                   linestyle=my_settings['midline_linestyle'],
                   alpha=my_settings['midline_alpha'])
    
    # Style the plot
    plt.grid(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    
    # Statistical testing and significance stars
    data = df_plot[value_col]
    
    # One-sample t-test against test_value
    t_stat, p_value = stats.ttest_1samp(data, test_value)
    
    # Determine significance stars and their fontsize
    sig_stars = significance_stars(p_value)
    if sig_stars in ("*", "**", "***"):
        star_fontsize = my_settings['star_fontsize']  
    else:
        star_fontsize = my_settings['star_fontsize_ns']
    
    # Calculate positions for significance indicators
    data_max = df_plot[value_col].max()
    if my_settings['ymax'] is not None:
        y_max = my_settings['ymax']
        y_min = my_settings['ymin']
    else:
        y_max = data_max * 1.1
        y_min = data_max * 1.1

    # Add significance stars above the plot
    star_height = y_max + (abs(y_max - y_min)) * 0.02
    plt.text(0, star_height, sig_stars, ha='center', va='center', 
            fontsize=star_fontsize, fontweight='bold')
    
    # Set y-limits
    if my_settings['ymin'] is not None and my_settings['ymax'] is not None:
        plt.ylim(my_settings['ymin'], star_height + (abs(y_max - y_min)) * 0.04 )
    
    # Print results in format: One-sample t-test against [test_value]: t-statistic: t_stat; p-value: p_value; Significance: sig_stars
    print(f"\nOne-sample t-test against {test_value}: t-statistic: {t_stat:.4f} p-value: {p_value:.4f} Significance: {sig_stars}\n")
    
    plt.tight_layout()

    # Tick formatting
    plt.tick_params(axis='y', which='major', length=6, 
                    width=1.5, left=True, direction='in')
    plt.tick_params(axis='both', labelsize=my_settings['fontsize'])

    return plt
