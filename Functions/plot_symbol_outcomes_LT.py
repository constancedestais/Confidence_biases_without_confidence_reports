
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_symbol_outcomes_LT(df_mean_per_participant):
    """"
    INPUTS:
    df_mean_per_participant: DataFrame containing the mean outcomes for each symbol per participant (already averaged over trials, per participant and per valence condition)
    """

    # Reshape data from wide to long format for easier plotting with seaborn
    df_long = pd.melt(df_mean_per_participant, 
                    id_vars=['participant_ID', 'is_gain_trial','symbol_1_expected_value','symbol_2_expected_value'],
                    value_vars=['symbol_1_outcome', 'symbol_2_outcome'],
                    var_name='symbol',
                    value_name='outcome')

    # Map numerical values to labels
    df_long['condition'] = df_long['is_gain_trial'].map({True: 'Gain', False: 'Loss'})
    df_long['symbol_label'] = df_long['symbol'].map({'symbol_1_outcome': 'Symbol 1', 'symbol_2_outcome': 'Symbol 2'})

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # Color palette
    palette = {"Symbol 1": "darkgrey", "Symbol 2": "lightgrey"}

    # add references for Gain and Loss symbols' expected values at y = 0.775, 0.325,  -0.775 and -0.325
    # compute expected values
    '''
    EV_symbol1_GAIN = df_mean_per_participant['symbol_1_probability_best_outcome'] * df_mean_per_participant['symbol_1_best_outcome'] + (1 - df_mean_per_participant['symbol_1_probability_best_outcome']) * df_mean_per_participant['symbol_1_worst_outcome']
    EV_symbol2_GAIN = df_mean_per_participant['symbol_2_probability_best_outcome'] * df_mean_per_participant['symbol_2_best_outcome'] + (1 - df_mean_per_participant['symbol_2_probability_best_outcome']) * df_mean_per_participant['symbol_2_worst_outcome']
    EV_symbol1_LOSS = df_mean_per_participant['symbol_1_probability_best_outcome'] * df_mean_per_participant['symbol_1_best_outcome'] + (1 - df_mean_per_participant['symbol_1_probability_best_outcome']) * df_mean_per_participant['symbol_1_worst_outcome']
    EV_symbol2_LOSS = df_mean_per_participant['symbol_2_probability_best_outcome'] * df_mean_per_participant['symbol_2_best_outcome'] + (1 - df_mean_per_participant['symbol_2_probability_best_outcome']) * df_mean_per_participant['symbol_2_worst_outcome']  

    '''
    # plot the expected values of symbols as a dotted line
    EV_symbol1_GAIN = df_mean_per_participant[df_mean_per_participant['is_gain_trial'] == True]['symbol_1_expected_value'].unique()
    EV_symbol2_GAIN = df_mean_per_participant[df_mean_per_participant['is_gain_trial'] == True]['symbol_2_expected_value'].unique()
    EV_symbol1_LOSS = df_mean_per_participant[df_mean_per_participant['is_gain_trial'] == False]['symbol_1_expected_value'].unique()
    EV_symbol2_LOSS = df_mean_per_participant[df_mean_per_participant['is_gain_trial'] == False]['symbol_2_expected_value'].unique()
    axes[0].axhline(y=EV_symbol1_GAIN, color='red', alpha=0.4, linewidth=1, linestyle='--', label='EV')
    axes[0].axhline(y=EV_symbol2_GAIN, color='red', alpha=0.4, linewidth=1, linestyle='--', label='EV')
    axes[1].axhline(y=EV_symbol2_LOSS, color='red', alpha=0.4, linewidth=1, linestyle='--', label='EV')
    axes[1].axhline(y=EV_symbol1_LOSS, color='red', alpha=0.4, linewidth=1, linestyle='--', label='EV')

    # temporarily plot original EV values    
    axes[0].axhline(y=0.775, color='black', alpha=0.2, linewidth=1, linestyle='--', label='EV')
    axes[0].axhline(y=0.325, color='black', alpha=0.2, linewidth=1, linestyle='--', label='EV')
    axes[1].axhline(y=-0.775, color='black', alpha=0.2, linewidth=1, linestyle='--', label='EV')
    axes[1].axhline(y=-0.325, color='black', alpha=0.2, linewidth=1, linestyle='--', label='EV')

    # Gain condition subplot (left)
    sns.boxplot(x='symbol_label', y='outcome', data=df_long[df_long['condition'] == 'Gain'],
                palette=palette, width=0.5, ax=axes[0])

    # Add individual data points for gain condition
    sns.stripplot(x='symbol_label', y='outcome', data=df_long[df_long['condition'] == 'Gain'],
                jitter=True, dodge=True, marker='o', alpha=0.5, color='darkgrey', ax=axes[0])

    # Loss condition subplot (right)
    sns.boxplot(x='symbol_label', y='outcome', data=df_long[df_long['condition'] == 'Loss'],
                palette=palette, width=0.5, ax=axes[1])

    # Add individual data points for loss condition
    sns.stripplot(x='symbol_label', y='outcome', data=df_long[df_long['condition'] == 'Loss'],
                jitter=True, dodge=True, marker='o', alpha=0.5, color='darkgrey', ax=axes[1])

    # Add titles and labels
    axes[0].set_title('Gain Condition', fontsize=14)
    axes[1].set_title('Loss Condition', fontsize=14)
    axes[0].set_xlabel('Symbol', fontsize=12)
    axes[1].set_xlabel('Symbol', fontsize=12)
    # Set the limits for each subplot
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(-1, 0)
    axes[0].set_ylabel('Outcome', fontsize=12)
    axes[1].set_ylabel('')  # No y-label for right subplot (shared y-axis)

    # Add a main title
    fig.suptitle('Symbol Outcomes in Gain and Loss in Learning Task', fontsize=16, y=1.05)

    stats = df_mean_per_participant.groupby(['participant_ID','is_gain_trial'])[['symbol_1_outcome', 'symbol_2_outcome']].agg(['mean']).reset_index()
    print(stats)

    return plt
