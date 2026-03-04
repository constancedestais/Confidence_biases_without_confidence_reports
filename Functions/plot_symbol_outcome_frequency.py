
# create function that plot_symbol_outcome_frequency
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_symbol_outcome_frequency(df_mean_per_participant):
    '''
    INPUTS:
    - df_mean_per_participant: DataFrame containing mean values of symbol outcomes per participant (of course per symbol id)
        - participant_ID
        - is_gain_trial
        _ symbol_id
        - outcome
        - frequency (proportion of times this outcome was received for this symbol, for this participant
    - EV_dict: dictionary containing expected values of symbols for gain and loss conditions (if available)
        - 'symbol_1_GAIN': EV_symbol1_GAIN
        - 'symbol_2_GAIN': EV_symbol2_GAIN
        - 'symbol_1_LOSS': EV_symbol1_LOSS
        - 'symbol_2_LOSS': EV_symbol2_LOSS

    '''

    df = df_mean_per_participant.copy()

    plt.figure(figsize=(12,6))

    # Colors + ordering
    custom_palette = {1:"darkgreen", 0.1:"green", -0.1:"red", -1:"darkred"}
    outcome_order = [-1, -0.1, 0.1, 1]
    x_order = sorted(df["symbol_id"].unique())

    #%% 1) Reference ticks BEHIND everything 
    # Aggregate to one reference value per group (use .first() if identical across rows)
    ref_df = (
        df.groupby(["symbol_id", "outcome"], as_index=False)["symbol_probability_best_outcome"]
        .mean()
    )
    # add a column expected_probability based on symbol_probability_best_outcome
    # for each symbol, if the outcome is the best outcome (1 or -0.1), then expected_probability = symbol_probability_best_outcome
    # for each symbol, if the outcome is the worst outcome (0.1 or -1), then expected_probability = 1-symbol_probability_best_outcome
    ref_df['expected_probability'] = ref_df.apply(lambda row: row['symbol_probability_best_outcome'] if row['outcome'] in [1, -0.1] else 1 - row['symbol_probability_best_outcome'], axis=1)

    # Overlay reference ticks (one per symbol × outcome)
    # marker "_" = short horizontal line; increase size for longer line
    ax = sns.stripplot(
        data=ref_df,
        x="symbol_id",
        y="expected_probability",
        hue="outcome",
        order=x_order,
        hue_order=outcome_order,
        dodge=True,
        jitter=False,
        marker="_",       # horizontal tick
        size=50,          # length of the tick
        linewidth=2,
        palette={o: "grey" for o in outcome_order},  # keep refs black (or use custom_palette)
        zorder=0,         # << behind
        legend=False
    )

    #%% 2) Boxplots on top

    # Boxplots (grouped by symbol, hue = outcome)
    ax = sns.boxplot(
        data=df,
        x="symbol_id",
        y="frequency",
        hue="outcome",
        order=x_order,
        hue_order=outcome_order,
        palette=custom_palette,
        showcaps=True,
        showfliers=False
    )

    #%% 3) Participant points on top of that

    sns.stripplot(
        data=df,
        x="symbol_id",
        y="frequency",
        hue="outcome",
        order=x_order,
        hue_order=outcome_order,
        dodge=True,
        jitter=True,
        alpha=0.4,
        palette=custom_palette,
        legend=False
    )



    #%% 4) legend and axis labels

    plt.xlabel("Symbol")
    plt.ylabel("Frequency")
    plt.title("Frequencies by Symbol and Outcome with Per-Outcome Reference")

    return plt