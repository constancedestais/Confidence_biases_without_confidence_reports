import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols


def _ensure_columns(df, required_cols):
    """Validate that df contains required_cols; raise ValueError if not."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in merged_data: {missing}")


def _safe_label_for_filename(label):
    """Make a model label safe to use in filenames."""
    return (
        label.replace("\n", " ")
             .replace("/", "-")
             .replace("\\", "-")
             .replace(":", "-")
             .replace("  ", " ")
             .strip()
    )

def set_axes_size_inch(fig, ax_list, width_in, height_in, left_in=1.0, bottom_in=0.8):
    """Set axes to an exact size in inches; figure may grow later for labels."""
    fig_w, fig_h = fig.get_size_inches()

    # Convert inches to figure-relative coordinates
    left = left_in / fig_w
    bottom = bottom_in / fig_h
    w = width_in / fig_w
    h = height_in / fig_h

    for ax in ax_list:
        ax.set_position([left, bottom, w, h])
        
def fit_models_save_summaries_and_ic(
    data,
    formulas,
    summary_dir,
    summary_prefix,
    print_summaries=True,
    store_fitted_results=False,
):
    """Fit OLS models, save summaries, and return a dict with AIC/BIC per model."""
    # Ensure the output directory exists.
    os.makedirs(summary_dir, exist_ok=True)

    results = {}

    # Fit each model and save summary text.
    for name, formula in formulas.items():
        # Fit OLS model from formula.
        model = ols(formula, data=data)
        res = model.fit()

        # Optionally print results for quick debugging/inspection.
        if print_summaries:
            print(res.summary())
            print(
                f"\n{name}, n={res.nobs}, k={len(res.params)}, "
                f"AIC={res.aic}, BIC={res.bic}, BIC-AIC={res.bic - res.aic}"
            )

        # Save model summary to a text file.
        safe_name = _safe_label_for_filename(name)
        summary_path = os.path.join(summary_dir, f"{summary_prefix}_{safe_name}.txt")
        with open(summary_path, "w") as f:
            f.write(res.summary().as_text())

        # Store only what we need for plotting/comparison (plus optional fitted object).
        results[name] = {
            "formula": formula,
            "AIC": float(res.aic),
            "BIC": float(res.bic),
            "nobs": float(res.nobs),
            "k_params": int(len(res.params)),
        }
        if store_fitted_results:
            results[name]["result"] = res

    return results

def plot_aic_bic_points_two_axes(
    results,
    title,
    outpath,
    width=0.35,  # kept for API compatibility (not used)
    figsize= (5, 3), #(4, 2.5),
    fontname="Arial",
    fontsize=8,
    pad_frac=0.05,
    connect_points=True,
    x_offset=0.06,
):
    """Plot AIC (right axis) and BIC (left axis) as points (optionally connected); save to outpath."""
    # Keep the insertion order of models (dict order in Python 3.7+).
    model_names = list(results.keys())

    # Extract AIC/BIC values in the same order as model_names.
    aic_values = [results[name]["AIC"] for name in model_names]
    bic_values = [results[name]["BIC"] for name in model_names]

    # X positions for models.
    x = np.arange(len(model_names))

    # Create figure with left axis (BIC) and right axis (AIC).
    fig, ax_bic = plt.subplots(figsize=figsize)
    ax_aic = ax_bic.twinx()

    # Slight offset so AIC/BIC points don't overlap exactly.
    x_bic = x - x_offset
    x_aic = x + x_offset

    # set colors 
    bic_color = "#922782ff" #"purple"
    aic_color = "#bb5816ff" # "orange"


    # Plot points (and optionally connect them).
    ls = "-" if connect_points else "None"
    ax_bic.plot(x_bic, bic_values, marker="o", linestyle=ls, label="BIC", color=bic_color)
    ax_aic.plot(x_aic, aic_values, marker="o", linestyle=ls, label="AIC", color=aic_color)

    # X-axis formatting.
    ax_bic.set_xticks(x)
    ax_bic.set_xticklabels(model_names, fontsize=8)
    ax_bic.tick_params(axis="x", direction="in")

    # Y-axis labels.
    ax_bic.set_ylabel("BIC", color=bic_color)
    ax_aic.set_ylabel("AIC", rotation=270, va="center", color=aic_color)

    # Compute axis limits so each axis starts near its own minimum.
    bic_min, bic_max = min(bic_values), max(bic_values)
    aic_min, aic_max = min(aic_values), max(aic_values)

    bic_pad = (bic_max - bic_min) * pad_frac if bic_max > bic_min else 1.0
    aic_pad = (aic_max - aic_min) * pad_frac if aic_max > aic_min else 1.0

    ax_bic.set_ylim(bic_min - bic_pad, bic_max + bic_pad)
    ax_aic.set_ylim(aic_min - aic_pad, aic_max + aic_pad)
    

    # Cosmetics (match your existing style).
    ax_bic.grid(False)

    for ax in (ax_bic, ax_aic):
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    # Tick widths
    ax_bic.tick_params(axis="both", width=0.6)
    ax_aic.tick_params(axis="y", width=0.6)

    # Y-axis tick colors + direction
    ax_bic.tick_params(axis="y", colors=bic_color, direction="in")
    ax_aic.tick_params(axis="y", colors=aic_color, direction="in")

    # Hide unused spines
    ax_bic.spines["top"].set_visible(False)
    ax_bic.spines["right"].set_visible(False)
    ax_bic.spines["bottom"].set_visible(False)

    ax_aic.spines["top"].set_visible(False)
    ax_aic.spines["left"].set_visible(False)
    ax_aic.spines["bottom"].set_visible(False)

    # Color the visible y-axis spines to match each series
    ax_bic.spines["left"].set_color(bic_color)
    ax_aic.spines["right"].set_color(aic_color)

    # X-axis ticks (labels only, no tick marks)
    ax_bic.tick_params(axis="x", which="both", bottom=False, top=False, length=0, labelbottom=True)

    # Apply font formatting.
    for ax in (ax_bic, ax_aic):
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(fontsize)
            item.set_fontname(fontname)

    # Annotate points with numeric values.
    def _label_points(axis, xs, ys, fontsize, color):
        for xi, yi in zip(xs, ys):
            axis.annotate(
                f"{yi:.1f}",
                xy=(xi, yi),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color=color,
            )

    _label_points(ax_bic, x_bic, bic_values, fontsize - 2, bic_color)
    _label_points(ax_aic, x_aic, aic_values, fontsize - 2, aic_color)

    # Keep exact axes size behavior from your original function.
    set_axes_size_inch(fig, [ax_bic, ax_aic], width_in=2.95, height_in=1.8)

    # Save and close.
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    plt.show()
    plt.close(fig)


def plot_aic_bic_bars_two_axes(
    results,
    title,
    outpath,
    width=0.35,
    figsize=(4, 2.5),
    fontname="Arial",
    fontsize=8,
    pad_frac=0.01,
):
    """Plot AIC (right axis) and BIC (left axis) as grouped bars; save to outpath."""
    # Keep the insertion order of models (dict order in Python 3.7+).
    model_names = list(results.keys())

    # Extract AIC/BIC values in the same order as model_names.
    aic_values = [results[name]["AIC"] for name in model_names]
    bic_values = [results[name]["BIC"] for name in model_names]

    # X positions for grouped bars.
    x = np.arange(len(model_names))

    # Create figure with left axis (BIC) and right axis (AIC).
    fig, ax_bic = plt.subplots(figsize=figsize)
    ax_aic = ax_bic.twinx()

    # Draw bars: AIC on right side, BIC on left side for each model group.
    rects_aic = ax_aic.bar(x + width / 2, aic_values, width, label="AIC", color="skyblue")
    rects_bic = ax_bic.bar(x - width / 2, bic_values, width, label="BIC", color="salmon")

    # X-axis formatting.
    ax_bic.set_xticks(x)
    ax_bic.set_xticklabels(model_names, fontsize=8)
    ax_bic.tick_params(axis="x", direction="in")

    # Y-axis labels (explicit which side is which).
    ax_bic.set_ylabel("BIC")
    ax_aic.set_ylabel("AIC", rotation=270, va="center")

    # Compute axis limits so each axis starts near its own minimum.
    bic_min, bic_max = min(bic_values), max(bic_values)
    aic_min, aic_max = min(aic_values), max(aic_values)

    # Small padding helps avoid bars/labels touching the axes.
    bic_pad = (bic_max - bic_min) * 0.05 if bic_max > bic_min else 1.0
    aic_pad = (aic_max - aic_min) * 0.05 if aic_max > aic_min else 1.0

    ax_bic.set_ylim(bic_min - bic_pad, bic_max + bic_pad)
    ax_aic.set_ylim(aic_min - aic_pad, aic_max + aic_pad)

    # Cosmetics: match your style choices.
    #ax_bic.set_title(title)
    ax_bic.grid(False)

    # Thinner axis lines (spines)
    for ax in (ax_bic, ax_aic):
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    # Thinner tick marks
    ax_bic.tick_params(axis="both", width=0.6)
    ax_aic.tick_params(axis="y", width=0.6)  # right axis ticks

    # Move ticks inward
    ax_bic.tick_params(axis="y", direction="in")
    ax_aic.tick_params(axis="y", direction="in")

    # Remove top spine on both; remove right spine only from the left axis
    ax_bic.spines["top"].set_visible(False)
    ax_aic.spines["top"].set_visible(False)
    ax_bic.spines["right"].set_visible(False)
    ax_aic.spines["left"].set_visible(False)
    ax_bic.spines["bottom"].set_visible(False)
    ax_aic.spines["bottom"].set_visible(False)

    # Keep model labels, remove x tick marks
    ax_bic.tick_params(axis="x", which="both", bottom=False, top=False, length=0, labelbottom=True)

    # Apply font formatting across both axes.
    for ax in (ax_bic, ax_aic):
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(fontsize)
            item.set_fontname(fontname)

    # Label each bar with its numeric value (on the correct axis).
    def _label_bars(axis, rects, fontsize):
        """Annotate bars with their heights."""
        for r in rects:
            h = r.get_height()
            axis.annotate(
                f"{h:.1f}",
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize
            )
    _label_bars(ax_aic, rects_aic, fontsize-2)
    _label_bars(ax_bic, rects_bic, fontsize-2)

    def _label_points(axis, xs, ys, fontsize):
        for xi, yi in zip(xs, ys):
            axis.annotate(
                f"{yi:.1f}",
                xy=(xi, yi),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize
            )
    _label_points(ax_bic, x_bic, bic_values, fontsize-2)
    _label_points(ax_aic, x_aic, aic_values, fontsize-2)


    # Combine legends from both axes into one.
    #h_bic, l_bic = ax_bic.get_legend_handles_labels()
    #h_aic, l_aic = ax_aic.get_legend_handles_labels()
    #ax_bic.legend(h_aic + h_bic, l_aic + l_bic, loc="best")

    # after plotting/labels are set
    set_axes_size_inch(fig, [ax_bic, ax_aic], width_in=2.95, height_in=1.8)

    # Save output and close the figure to avoid memory leaks in loops.
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    plt.show()
    plt.close(fig)






def plot_regressions_model_comparison_RLtreatment_CFCtreatment(
    merged_data,
    independent_variable,
    version_code,
):
    """Compare RL+CFC candidate models via AIC/BIC and save plot + summaries."""
    # Check required columns.
    required_cols = [
        independent_variable,
        "CFC_chose_highest_expected_value",
        "click_desired",
        "identify_best",
        "participant_ID",
        "exp_ID",
    ]
    _ensure_columns(merged_data, required_cols)

    # Define models to compare.
    formulas = {
        "NULL": f"CFC_chose_highest_expected_value ~ {independent_variable}",
        "RL_x": (
            f"CFC_chose_highest_expected_value ~ {independent_variable} + "
            f"C(click_desired)"
        ),
        "CFC_x": (
            f"CFC_chose_highest_expected_value ~ {independent_variable} + "
            f"C(identify_best)"
        ),
        "ADD": (
            f"CFC_chose_highest_expected_value ~ {independent_variable} + "
            f"C(click_desired) + C(identify_best)"
        ),
        "FULL": (
            f"CFC_chose_highest_expected_value ~ {independent_variable} + "
            f"C(click_desired) + C(identify_best) + "
            f"C(identify_best):C(click_desired)"
        ),
    }

    # Fit models and save summaries.
    results = fit_models_save_summaries_and_ic(
        data=merged_data.copy(),
        formulas=formulas,
        summary_dir="./Outputs/Figures",
        summary_prefix=f"regression_model_{independent_variable}_RLtreatment_CFCtreatment_{version_code}",
        print_summaries=True,
        store_fitted_results=False,
    )

    # Plot AIC/BIC with two y-axes (BIC left, AIC right).
    ''' 
    plot_aic_bic_bars_two_axes(
        results=results,
        title="Model Comparison using AIC and BIC",
        outpath=f"./Outputs/Figures/regression_model_comparison_RLtreatment_CFCtreatment_{independent_variable}_{version_code}.svg",
        width=0.38,      # keep your preferred width here
        pad_frac=0.01,   # keep your preferred padding here
    )
    '''
    plot_aic_bic_points_two_axes(
        results=results,
        title="Model Comparison using AIC and BIC",
        outpath=f"./Outputs/Figures/regression_model_comparison_RLtreatment_CFCtreatment_{independent_variable}_{version_code}.svg",
    )
    a=1







def plot_regressions_model_comparison_difficulty_CFCtreatment(
    merged_data,
    independent_variable,
    version_code,
):
    """Compare difficulty+CFC candidate models via AIC/BIC and save plot + summaries."""
    # Check required columns.
    required_cols = [
        independent_variable,
        "CFC_chose_highest_expected_value",
        "click_desired",
        "identify_best",
        "LT_unequal_difficulty_binary",
        "participant_ID",
        "exp_ID",
    ]
    _ensure_columns(merged_data, required_cols)

    # Define models to compare.
    formulas = {
        "NULL": f"CFC_chose_highest_expected_value ~ {independent_variable}",
        "ASYM_x": (
            f"CFC_chose_highest_expected_value ~ {independent_variable} + "
            f"C(LT_unequal_difficulty_binary)"
        ),
        "CFC_x": (
            f"CFC_chose_highest_expected_value ~ {independent_variable} + "
            f"C(identify_best)"
        ),
        "ADD": (
            f"CFC_chose_highest_expected_value ~ {independent_variable} + "
            f"C(identify_best) + C(LT_unequal_difficulty_binary)"
        ),
        "FULL": (
            f"CFC_chose_highest_expected_value ~ {independent_variable} + "
            f"C(identify_best) + C(LT_unequal_difficulty_binary) + "
            f"C(identify_best):C(LT_unequal_difficulty_binary)"
        ),

    }

    # Fit models and save summaries.
    results = fit_models_save_summaries_and_ic(
        data=merged_data.copy(),
        formulas=formulas,
        summary_dir="./Outputs/Figures",
        summary_prefix=f"regression_model_{independent_variable}_difficulty_CFCtreatment_{version_code}",
        print_summaries=True,
        store_fitted_results=False,
    )

    # Plot AIC/BIC with two y-axes (BIC left, AIC right).
    '''
    plot_aic_bic_bars_two_axes(
        results=results,
        title=(
            "Model Comparison using AIC and BIC\n"
            f"Model: CFC_chose_highest_expected_value ~ {independent_variable} "
            "with variables: difficulty manipulation & identify_best"
        ),
        outpath=f"./Outputs/Figures/regression_model_comparison_difficulty_CFCtreatment_{independent_variable}_{version_code}.svg",
        width=0.35,
        pad_frac=0.01,
    )
 
    '''
    plot_aic_bic_points_two_axes(
        results=results,
        title=(
                    "Model Comparison using AIC and BIC\n"
                    f"Model: CFC_chose_highest_expected_value ~ {independent_variable} "
                    "with variables: difficulty manipulation & identify_best"
                ),
        outpath=f"./Outputs/Figures/regression_model_comparison_difficulty_CFCtreatment_{independent_variable}_{version_code}.svg",
    )

    



def plot_regressions_model_comparison_CFCtreatment(
    merged_data,
    independent_variable,
    version_code,
):
    """Compare NULL vs CFC-treatment model via AIC/BIC and save plot + summaries."""
    # Check required columns.
    required_cols = [
        independent_variable,
        "CFC_chose_highest_expected_value",
        "click_desired",
        "identify_best",
        "participant_ID",
        "exp_ID",
    ]
    _ensure_columns(merged_data, required_cols)

    # Define models to compare.
    formulas = {
        "NULL": f"CFC_chose_highest_expected_value ~ {independent_variable}",
        "CFC_x": (
            f"CFC_chose_highest_expected_value ~ {independent_variable} + "
            f"C(identify_best)"
        ),
    }

    # Fit models and save summaries.
    results = fit_models_save_summaries_and_ic(
        data=merged_data.copy(),
        formulas=formulas,
        summary_dir="./Outputs/Figures",
        summary_prefix=f"regression_model_{independent_variable}_CFCtreatment_{version_code}",
        print_summaries=True,
        store_fitted_results=False,
    )

    # Plot AIC/BIC with two y-axes (BIC left, AIC right).
    '''
    plot_aic_bic_bars_two_axes(
        results=results,
        title=(
            "Model Comparison using AIC and BIC\n"
            f"Model: CFC_chose_highest_expected_value ~ {independent_variable} "
            "with variables: identify_best"
        ),
        outpath=f"./Outputs/Figures/regression_model_comparison_CFCtreatment_{independent_variable}_{version_code}.svg",
        width=0.35,
        pad_frac=0.01,
    )

    '''
    plot_aic_bic_points_two_axes(
        results=results,
        title=(
            "Model Comparison using AIC and BIC\n"
            f"Model: CFC_chose_highest_expected_value ~ {independent_variable} "
            "with variables: identify_best"
        ),
        outpath=f"./Outputs/Figures/regression_model_comparison_CFCtreatment_{independent_variable}_{version_code}.svg",
    )

