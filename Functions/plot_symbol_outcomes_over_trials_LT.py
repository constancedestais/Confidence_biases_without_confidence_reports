import matplotlib.pyplot as plt
import seaborn as sns

def plot_symbol_outcomes_over_trials_LT(df_mean_per_participant_per_trial):

    #%% set up

    # set the style of seaborn
    sns.set_theme(style="whitegrid")
    # set the figure size
    plt.figure(figsize=(12, 6))
    # set the color palette
    palette = sns.color_palette("husl", 8)

    # Create a figure with a reasonable size
    plt.figure(figsize=(12, 8))

    # Variables to plot
    variables_to_plot = [
        'symbol_1_outcome', 
        'symbol_2_outcome'
    ]

    # Define colors for each variable
    valence_colors = {
        True: 'green',    # circle for gain trials
        False: 'red' 
    }

     # Define line colors for each variable - will be concateded with valence colors
    variable_line_colors = {
        'symbol_1_outcome': 'dark',
        'symbol_2_outcome': '',
    }

    ''''
    # Define line styles for gain/loss trials
    variable_line_styles = {
        'symbol_1_outcome': '--',
        'symbol_2_outcome': ':',
    }

    # Define markers for gain/loss trials
    markers = {
        True: '+',    # circle for gain trials
        False: 'x'    # x for loss trials
    }
    '''


    #%% plotting

    # Loop through the variables to plot
    for var in variables_to_plot:
        # Group by trial number and gain/loss, calculating the mean across participants
        for is_gain, group in df_mean_per_participant_per_trial.groupby('is_gain_trial'):
            # Calculate mean values for each trial number for this gain/loss group
            trial_means = group.groupby('trial_per_pair')[var].mean().reset_index()
            
            # color gets name from concatenation of valence_colors and variable_line_colors
            current_color = variable_line_colors[var] + valence_colors[is_gain]

            # Plot this specific variable for this gain/loss condition
            plt.plot(
                trial_means['trial_per_pair'], 
                trial_means[var], 
                color=current_color,
                alpha=0.7,  # Adjust transparency for individual lines
                #linestyle=variable_line_styles[var],
                marker = '+', #marker=markers[is_gain],
                label=f"{var.replace('_', ' ')} ({'Gain' if is_gain else 'Loss'})"
            )
            '''
            # plot the mean line for each variable
            plt.axhline(y=trial_means[var].mean(),
                        color=valence_colors[is_gain], 
                        alpha=0.2, 
                        #linestyle=variable_line_styles[var],
                        linewidth = 3
            )
            '''

    # plot the expected values of symbols as a dotted red line
    EV_symbol1_GAIN = df_mean_per_participant_per_trial[df_mean_per_participant_per_trial['is_gain_trial'] == True]['symbol_1_expected_value'].unique()
    EV_symbol2_GAIN = df_mean_per_participant_per_trial[df_mean_per_participant_per_trial['is_gain_trial'] == True]['symbol_2_expected_value'].unique()
    EV_symbol1_LOSS = df_mean_per_participant_per_trial[df_mean_per_participant_per_trial['is_gain_trial'] == False]['symbol_1_expected_value'].unique()
    EV_symbol2_LOSS = df_mean_per_participant_per_trial[df_mean_per_participant_per_trial['is_gain_trial'] == False]['symbol_2_expected_value'].unique()
    plt.axhline(y=EV_symbol1_GAIN,linewidth = 1 , color='red', alpha=0.4, linestyle='--')
    plt.axhline(y=EV_symbol2_GAIN,linewidth = 1 , color='red', alpha=0.4, linestyle='--')
    plt.axhline(y=EV_symbol1_LOSS,linewidth = 1 , color='red', alpha=0.4, linestyle='--')
    plt.axhline(y=EV_symbol2_LOSS,linewidth = 1 , color='red', alpha=0.4, linestyle='--')

    
    plt.axhline(y=0.775,linewidth = 1 , color='black', alpha=0.2, linestyle='--')
    plt.axhline(y=0.325,linewidth = 1 , color='black', alpha=0.2, linestyle='--')
    plt.axhline(y=-0.775,linewidth = 1 , color='black', alpha=0.2, linestyle='--')
    plt.axhline(y=-0.325,linewidth = 1 , color='black', alpha=0.2, linestyle='--')
    

    # Add a second legend for the line styles
    legend1 = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().add_artist(legend1)

    # Create custom line objects for the second legend (gain/loss)
    '''
    gain_line = plt.Line2D([0], [0], color='black', linestyle='-', marker='o', label='Gain')
    loss_line = plt.Line2D([0], [0], color='black', linestyle='--', marker='x', label='Loss')
    plt.legend(handles=[gain_line, loss_line], bbox_to_anchor=(1.05, 0.7), loc='upper left')
    '''

    # Add plot details
    plt.xlabel('Trial Number', fontsize=12)
    plt.ylabel('Outcome', fontsize=12)
    plt.title('Symbol Outcomes by Trial Number in Learning Task', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()



    return plt