import matplotlib.pyplot as plt
import seaborn as sns

def plot_symbol_probabilities_over_trials_LT(df_mean_per_participant_per_trial):
    
    #%% Validate dataset structure: Check for exactly one entry per participant, per trial, per condition
    validation_counts = df_mean_per_participant_per_trial.groupby(['participant_ID', 'trial_per_pair', 'is_gain_trial']).size().reset_index(name='count')
    # Assert statement to check that all combinations have exactly one entry
    try:
        assert (validation_counts['count'] == 1).all(), "Error: Some participant-trial-condition combinations have duplicate entries"
        print("Dataset validation passed: One entry per participant per trial per condition")
    except AssertionError as e:
        print(e)
        # Find problematic entries
        problem_entries = validation_counts[validation_counts['count'] > 1]
        print("Problematic entries (duplicates):")
        print(problem_entries)
        # You might want to raise an exception here or handle the error appropriately
        raise ValueError("Dataset validation failed - see above for details")

    #%% plot settings

    # set the style of seaborn
    sns.set_theme(style="whitegrid")
    # set the figure size
    plt.figure(figsize=(12, 6))
    # set the color palette
    palette = sns.color_palette("husl", 8)
    
    # Create a figure with a reasonable size
    plt.figure(figsize=(12, 8))

    # Variables to plot
    variables_to_plot = ['symbol_1_probability_best_outcome', 'symbol_2_probability_best_outcome']

    # Define colors for each variable
    colors = {
        True: 'green',   
        False: 'red' 
    }

    # Define line styles for gain/loss trials
    line_styles = {
        'symbol_1_probability_best_outcome': '-',
        'symbol_2_probability_best_outcome': ':',
    }

    # Define markers for gain/loss trials
    markers = {
        True: '+',    # circle for gain trials
        False: 'x'    # x for loss trials
    }

    #%% plot 

    # Loop through the variables to plot
    for var in variables_to_plot:
        # Group by trial number and gain/loss, calculating the mean across participants
        for is_gain, group in df_mean_per_participant_per_trial.groupby('is_gain_trial'):
            # Calculate mean values for each trial number for this gain/loss group
            trial_means = group.groupby('trial_per_pair')[var].mean().reset_index()
                    
            # Plot this specific variable for this gain/loss condition
            plt.plot(
                trial_means['trial_per_pair'], 
                trial_means[var], 
                color=colors[is_gain],
                alpha=0.5,  # Adjust transparency for individual lines
                linestyle=line_styles[var],
                marker=markers[is_gain],
                label=f"{var.replace('_', ' ')} ({'Gain' if is_gain else 'Loss'})"
            )

    # Add plot details
    plt.xlabel('Trial Number', fontsize=8)
    plt.ylabel('Probability', fontsize=8)
    plt.title('Symbol P(best outcome) by Trial Number in Learning Task', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return plt
