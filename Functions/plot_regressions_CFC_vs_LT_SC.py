from pyexpat import model
import seaborn as sns
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, rlm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from patsy import dmatrix
from scipy.stats import norm
import matplotlib

from Functions.significance_stars import significance_stars 
#matplotlib.use("Agg")  # non-interactive backend
matplotlib.use("TkAgg")  # or "Qt5Agg", depending on what you have installed
import matplotlib.pyplot as plt
from Functions.set_axes_size import set_axes_size_inches
from Functions.my_colors import load_my_colors
from Functions.custom_barplot import custom_barplot


#Plot correlation between pChose Gain in CFC and (1) pCorrect[gain-loss] in LT and (2) pCorrect[gain-loss] in SC, and then show regression line 
#INPUTS
#- dataframe containing
#    - 'CFC_chose_highest_expected_value' column (pChose Gain in CFC)
#    - independent_variable representing accuracy[gain-loss] in LT: 'LT_correct_gain_minus_loss' or in SC: 'SC_correct_gain_minus_loss'
#    - participantID
#- combine_all_versions: boolean, whether to do regression over all exp_versions combined (True) or separately (False)
#- dependent independent_variable, present in merged_data: string, either 'LT_correct_gain_minus_loss' or 'SC_correct_gain_minus_loss'

#%% ---------- helper functions ----------

def _check_inputs(
    merged_data: pd.DataFrame,
    independent_variable: str,
    ):

    data = merged_data.copy()

    # check that required columns are present
    required_columns = ['CFC_chose_highest_expected_value',independent_variable,'exp_ID','participant_ID']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"DataFrame must contain the column: {col}")

    # check that IV and DV are in percentage (between 0 and 100) - so that max is higher than 1.5 but not higher than 100
    assert data['CFC_chose_highest_expected_value'].max() > 1.5 and data['CFC_chose_highest_expected_value'].max() <= 100, "CFC_chose_highest_expected_value should be in percentage (between 0 and 100)"
    assert data[independent_variable].max() > 1.5 and data[independent_variable].max() <= 100, f"{independent_variable} should be in percentage (between 0 and 100)"

    return data


def _get_expID_colors(merged_data: pd.DataFrame):
    data = merged_data.copy()

    # define colors for each exp_version
    expID_colors = {
        'cd1_2025_click_desired_1_identify_best_1': 'black',
        'cd1_2025_click_desired_1_identify_best_0': 'red',
        'cd1_2025_click_desired_0_identify_best_1': 'blue',
        'cd1_2025_click_desired_0_identify_best_0': 'purple',
        'cd1_2025_click_desired_1_identify_best_1_difficulty_0_70_0_80': 'grey',
        'cd1_2025_click_desired_1_identify_best_0_difficulty_0_70_0_80': 'orange',
        'all': 'black',
    }

    # check that all exp_versions present in dataset have a defined color
    exp_versions = data['exp_ID'].unique().tolist()
    for exp_version in exp_versions:
        if exp_version != 'all':
            if exp_version not in expID_colors:
                raise ValueError(f"No color defined for exp_version: {exp_version}")
        if exp_version not in expID_colors:
            raise ValueError(f"No color defined for exp_version: {exp_version}")
        
    # create new dictionary with only exp_versions present in data (exp_versions)
    expID_colors = {k: v for k, v in expID_colors.items() if k in exp_versions}
    # check lengths match
    if len(expID_colors) != len(exp_versions):
        raise ValueError("Mismatch between number of exp_versions in data and number of defined colors.")
    
    return expID_colors
        
def _run_regression_on_each_dataset(
    merged_data: pd.DataFrame,
    independent_variable: str,
    formula: str = None,
    ):
    # function runs regression on each dataset in the data, and extracts slopes and intercepts for each dataset

    data = merged_data.copy()

    # get unique exp_versions 
    exp_versions = data['exp_ID'].unique().tolist()

    # store intercept, intercept CI, intercept p-value AND store slope, slope CI, slope p-value
    models, intercepts, slopes, intercepts_ci, slopes_ci, intercepts_p_values, slopes_p_values, intercepts_err, slopes_err = [], [], [], [], [], [], [], [], []

    # Loop through each experiment version
    for exp_version in exp_versions:
        if exp_version == 'all':
            df_version = data.copy()
        else:
            df_version = data[data['exp_ID'] == exp_version].copy()

        model = ols(formula, data=df_version).fit()
        
        # store whole model states in an array of size n_exp_versions
        models.append(model)

        # extract slope, intercept, CI, p-values
        (intercept, intercept_ci, p_value_intercept, slope, slope_ci, p_value_slope) = _extract_slope_intercepts_p_values_from_regression( my_model = model, 
                                                                                                                                            independent_variable = independent_variable,
                                                                                                                                            intercept_expected_baseline = 50)
        # store outputs in list for each ex
        intercepts.append(intercept)
        intercepts_ci.append(intercept_ci)
        intercepts_p_values.append(p_value_intercept)
        slopes.append(slope)
        slopes_ci.append(slope_ci)
        slopes_p_values.append(p_value_slope)

    # Convert lists to numpy arrays
    intercepts = np.array(intercepts)
    slopes = np.array(slopes)
    intercepts_ci = np.array(intercepts_ci)
    slopes_ci = np.array(slopes_ci)
    intercepts_p_values = np.array(intercepts_p_values)
    slopes_p_values = np.array(slopes_p_values)     

    # check size of all output variables
    assert intercepts.shape[0] == len(exp_versions)
    assert slopes.shape[0] == len(exp_versions)
    assert intercepts_ci.shape[0] == len(exp_versions)
    assert slopes_ci.shape[0] == len(exp_versions)
    assert intercepts_p_values.shape[0] == len(exp_versions)
    assert slopes_p_values.shape[0] == len(exp_versions)

    return (models, 
            intercepts, 
            intercepts_ci, 
            intercepts_p_values, 
            slopes, 
            slopes_ci, 
            slopes_p_values)

def _extract_slope_intercepts_p_values_from_regression(my_model, 
                                                       independent_variable: str,
                                                       intercept_expected_baseline: float):
    # Function extracts slope, intercept and 95 % CI, compute p-value for intercept vs intercept_expected_baseline, and slope p-value vs 0 
    

    # Extract coefficients and 95 % CI
    ci = my_model.conf_int(alpha=0.05)
    intercept = my_model.params['Intercept']
    slope = my_model.params[independent_variable]
    intercept_ci = ci.loc['Intercept'].to_list()
    slope_ci = ci.loc[independent_variable].to_list()

    # --- Compute p-value for intercept vs 50 (like MATLAB) ---
    b0 = my_model.params['Intercept']
    se_b0 = my_model.bse['Intercept']
    df_resid = my_model.df_resid
    t_stat = (b0 - intercept_expected_baseline) / se_b0
    p_value_intercept = 2 * (1 - stats.t.cdf(abs(t_stat), df_resid))

    # slope p-value (vs 0)
    p_value_slope = my_model.pvalues[independent_variable]

    return (intercept, 
            intercept_ci, 
            p_value_intercept, 
            slope,
            slope_ci, 
            p_value_slope)



def _ols_mean_ci(result, new_data, rhs_formula, alpha=0.05,
                mean_col="y_hat", lower_col="ci_low", upper_col="ci_high"):
    
    '''
    Compute confidence intervals for the *mean* prediction from a statsmodels OLS result.
    inputs:
    - result: fitted OLS model (from smf.ols(...).fit())
    - new_data: DataFrame containing the predictor values to predict on. Must have all variables needed by `rhs_formula`.
    - rhs_formula: Patsy formula for the fixed-effects *right-hand side*, e.g. "1 + x + C(identify_best)".
    - alpha: significance level (default 0.05 for 95% CI)
    - mean_col, lower_col, upper_col: column names to use for mean and CI bounds in the returned DataFrame.
    returns:
    - pred_df: Copy of `new_data` with added columns:
        - mean_col  : mean prediction
        - lower_col : lower CI bound
        - upper_col : upper CI bound
    '''
    params = result.params
    cov_params = result.cov_params().loc[params.index, params.index]

    X = dmatrix(rhs_formula, new_data, return_type="dataframe")
    X = X.reindex(columns=params.index, fill_value=0.0)

    mean_pred = np.asarray(X @ params)
    var_pred = np.einsum("ij,jk,ik->i", X.values, cov_params.values, X.values)
    se_pred = np.sqrt(var_pred)

    z = norm.ppf(1 - alpha / 2)
    ci_low = mean_pred - z * se_pred
    ci_high = mean_pred + z * se_pred

    pred_df = new_data.copy()
    pred_df[mean_col] = mean_pred
    pred_df[lower_col] = ci_low
    pred_df[upper_col] = ci_high
    return pred_df

        

def _mixedlm_mean_ci(result, new_data, rhs_formula, alpha=0.05,
                    mean_col="y_hat", lower_col="ci_low", upper_col="ci_high"):
    """
    Compute confidence intervals for the *mean* prediction (fixed effects only)
    from a statsmodels MixedLM result.

    Parameters
    ----------
    result : statsmodels.regression.mixed_linear_model.MixedLMResults
        Fitted mixed model (from smf.mixedlm(...).fit()).

    new_data : pandas.DataFrame
        DataFrame containing the predictor values to predict on.
        Must have all variables needed by `rhs_formula`.

    rhs_formula : str
        Patsy formula for the fixed-effects *right-hand side*,
        e.g. "1 + x + C(identify_best)".

    alpha : float, optional
        Significance level (default 0.05 for 95% CI).

    mean_col, lower_col, upper_col : str
        Column names to use for mean and CI bounds in the returned DataFrame.

    Returns
    -------
    pred_df : pandas.DataFrame
        Copy of `new_data` with added columns:
        - mean_col  : mean prediction (fixed effects only)
        - lower_col : lower CI bound
        - upper_col : upper CI bound
    """
    # Fixed-effect parameters and covariance
    fe_params = result.fe_params
    cov_fe = result.cov_params().loc[fe_params.index, fe_params.index]

    # Design matrix for fixed effects
    X = dmatrix(rhs_formula, new_data, return_type="dataframe")

    # Mean predictions
    mean_pred = np.asarray(X @ fe_params)

    # Var( Xβ ) = diag( X Σβ Xᵀ )
    var_pred = np.einsum("ij,jk,ik->i", X.values, cov_fe.values, X.values)
    se_pred = np.sqrt(var_pred)

    z = norm.ppf(1 - alpha / 2)

    ci_low = mean_pred - z * se_pred
    ci_high = mean_pred + z * se_pred

    pred_df = new_data.copy()
    pred_df[mean_col] = mean_pred
    pred_df[lower_col] = ci_low
    pred_df[upper_col] = ci_high

    return pred_df

#%% ---------- plot_regression_of_LT_accuracy against Transfer accuracy ----------
def plot_regression_of_LT_accuracy_against_Transfer_accuracy(
        merged_data: pd.DataFrame,
        filenames: list):
    
    data = merged_data.copy()

    # check that values are in percentage (between 0 and 100) - so that max is higher than 1.5 but not higher than 100
    assert data['LT_correct_gain_minus_loss'].max() > 1.5 and data['LT_correct_gain_minus_loss'].max() <= 100, "LT_correct_gain_minus_loss should be in percentage (between 0 and 100)"
    assert data['SC_correct_gain_minus_loss'].max() > 1.5 and data['SC_correct_gain_minus_loss'].max() <= 100, "SC_correct_gain_minus_loss should be in percentage (between 0 and 100)"

    # reminder: winning_model =  f'CFC_chose_highest_expected_value ~ {independent_variable} + C(identify_best)' ++ mixed effects for participant_ID  

    # create figure
    fig, ax = plt.subplots(figsize=(4,4))
    fig.show()

    ax = sns.regplot(
        x=data['LT_correct_gain_minus_loss'],
        y=data['SC_correct_gain_minus_loss'],
        scatter_kws={'s': 50, 'color': 'black', 'alpha':    0.3},  # Customize scatter points
        line_kws={'color': 'black', 'linewidth': 2},  # Customize regression line
    )
    # figure settings
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('LT accuracy gain minus loss (%)', fontsize=10)
    ax.set_ylabel('SC accuracy gain minus loss (%)', fontsize=10)
    ax.set_title('Regression of accuracy [gain-loss] in LT against SC', fontsize=10)
    ax.tick_params(axis='both', labelsize=10)
    # change spine width
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(0.5)
    # change tick width
    ax.tick_params(width=0.5)
    # save figure
    fig.tight_layout()
    fig.savefig(filenames[0], dpi=300, bbox_inches="tight")
    plt.close()

     
    # plot parameter estimate for intercept and slope with bar plot
    formula = f"SC_correct_gain_minus_loss ~ LT_correct_gain_minus_loss"
    model = smf.ols(formula, data=data)
    result = model.fit()
    intercept = result.params['Intercept']
    intercept_ci = result.conf_int().loc['Intercept'].to_list()
    p_value_intercept = result.pvalues['Intercept']
    ref=0,
    y_ticks=[-10,0,10,15]
    y_lim=[-10,15]

    c = load_my_colors()
    figure_size = (3, 2)
    font_size = 8
    font_name="Arial"  # width, height in inches
    axis_linewidth = 0.75
    line_width = 0.7
    color = c['light_gray']
    error_capsize = 5
    width = 0.5
    star_fontsize = 12
    ns_fontsize = 8
    x_tick_labels = ['']
    figure_size = (0.9, 0.7) # width, height in inches

    # make bar plot
    fig, ax = custom_barplot(
                x = ['all'],
                y = [intercept],
                y_ci = [intercept_ci],
                p_values = [p_value_intercept],
                y_reference_value = ref,
                y_label = "intercept",
                y_ticks = y_ticks,
                x_tick_labels = x_tick_labels,
                y_lim = y_lim,
                error_capsize = error_capsize,
                color = color,
                width = width,
                star_fontsize = star_fontsize,
                ns_fontsize = ns_fontsize,
                font_name = font_name,
                font_size = font_size,
                axis_linewidth = axis_linewidth,
                figure_size = figure_size,
            )
    # change spine width
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(0.5)
    # change tick width
    ax.tick_params(width=0.5)
    # save figure
    fig.tight_layout()
    fig.savefig(filenames[1], dpi=300, bbox_inches="tight")
    



#%% ---------- plot_regression_over_all_data_residuals ----------
def plot_regression_over_all_data_residuals(
        merged_data: pd.DataFrame,
        independent_variable: str,
        filename: str):
    
    data = merged_data.copy()
    # check that necessary variables are present and between 0 and 100
    data = _check_inputs(merged_data=data, independent_variable=independent_variable)
    
    # reminder: winning_model =  f'CFC_chose_highest_expected_value ~ {independent_variable} + C(identify_best)' 

    # remove variance from CFC_chose_highest_expected_value and from independent_variable due to click_desired and identify_best, and plot residuals against each other
    formula1 = f'CFC_chose_highest_expected_value ~ C(identify_best)'
    model1 = ols(formula1, data=data) # linear regression, no mixed effects
    result1 = model1.fit()
    print(result1.summary())
    residuals1 = result1.resid

    formula2 = f"{independent_variable} ~ C(identify_best)"
    model2 = ols(formula2, data=data) # linear regression, no mixed effects
    result2 = model2.fit()
    print(result2.summary())
    residuals2 = result2.resid

    # plot residuals against each other
    # CAREFUL: this will always center the slope around 0, giving a better estimate of the slope, but not the intercept (which is now meaningless)
    # create figure
    fig, ax = plt.subplots(figsize=(4,4))
    fig.show()

    ax = sns.regplot(
        x=residuals2,
        y=residuals1,
        scatter_kws={'s': 50, 'color': 'black', 'alpha':    0.3},  # Customize scatter points
        line_kws={'color': 'black', 'linewidth': 2},  # Customize regression line
    )
    # figure settings
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel(f'Residuals of {independent_variable} (%)', fontsize=10)
    ax.set_ylabel('Residuals of pChose Gain in CFC (%)', fontsize=10)
    ax.set_title('Regression of residuals after regressing X (and Y) on click_desired and identify_best', fontsize=10)
    ax.tick_params(axis='both', labelsize=10)
    # change spine width
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(0.5)
    # change tick width
    ax.tick_params(width=0.5)
    # save figure
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

#%% ---------- plot_simple_regression_split_data_by_identify_best -------- 
def plot_simple_regression_split_data_by_identify_best(
    merged_data: pd.DataFrame,
    independent_variable: str,
    filename: str):
    
    data = merged_data.copy()
    # check that necessary variables are present and between 0 and 100
    data = _check_inputs(merged_data=data, independent_variable=independent_variable)
    
    # run and plot regression CFC_chose_highest_expected_value ~ {independent_variable} for identify_best = 0 and identify_best = 1 separately
    data_identify_best_0 = data[data['identify_best'] == 0].copy()
    data_identify_best_1 = data[data['identify_best'] == 1].copy()
    # print number of data points
    print(f'Number of data points with identify_best = 0: {len(data_identify_best_0)} (there should be 3x50=150)')

    c = load_my_colors()
    color_dict = {0: c['medium_gray'], 1: c['dark_gray']}

    fig, ax = plt.subplots(figsize=(4,4))
    fig.show()
    # plot identify_best = 0
    ax = sns.regplot(
        x=data_identify_best_0[independent_variable],
        y=data_identify_best_0['CFC_chose_highest_expected_value'],
        line_kws={'color': color_dict[0], 'linewidth': 2},  # Customize regression line
        scatter = False,
        label='identify_best = 0'
    )
    ax.scatter(
        data_identify_best_0[independent_variable],
        data_identify_best_0['CFC_chose_highest_expected_value'],
        s=25,
        facecolor=color_dict[0],
        alpha=0.3,
    )
    # plot identify_best = 1
    ax = sns.regplot(
        x=data_identify_best_1[independent_variable],
        y=data_identify_best_1['CFC_chose_highest_expected_value'],
        line_kws={'color': color_dict[1], 'linewidth': 2},  # Customize regression line
        scatter = False,
        label='identify_best = 1',
    )
    ax.scatter(
        data_identify_best_1[independent_variable],
        data_identify_best_1['CFC_chose_highest_expected_value'],
        s=25,
        facecolor=color_dict[1],
        alpha=0.3,
    )
    ax.legend(loc="best")
    # add line for x = 0 and y = 50
    ax.axhline(50, color='grey', linestyle='--', linewidth=1)
    ax.axvline(0, color='grey', linestyle='--', linewidth=1)
    # change spine width
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(0.5)
    # change tick width
    ax.tick_params(width=0.5)
    # save figure
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

#%% ---------- plot_simple_regression_parameters_across_datasets ---------- 
def plot_simple_regression_parameters_across_datasets(
    merged_data: pd.DataFrame,
    independent_variable: str,
    filenames: str,
    ):

    # will plot the intercepts for each dataset on one figure, and slopes on another figure

    data = merged_data.copy()

    # get exp_versions
    exp_versions = data['exp_ID'].unique().tolist()
    # check that necessary variables are present and between 0 and 100
    data = _check_inputs(merged_data=data, independent_variable=independent_variable)
    
    # simplest regression
    formula = f"CFC_chose_highest_expected_value ~ {independent_variable}"
    # run regression on each dataset and extract slopes and intercepts
    (models, 
    intercepts, 
    intercepts_ci, 
    intercepts_p_values, 
    slopes, 
    slopes_ci, 
    slopes_p_values) = _run_regression_on_each_dataset(
                            merged_data=data,
                            independent_variable=independent_variable,
                            formula=formula,
                        ) 
    
    # prepare variables for plotting
    parameter_label = ['Intercept','Slope']
    parameter = [intercepts,slopes]
    parameter_ci = [intercepts_ci,slopes_ci]
    parameter_p_values = [intercepts_p_values,slopes_p_values]  
    parameter_reference_value = [50,0]
    y_ticks= [[50, 65, 80],(0,0.5,1)]
    x_tick_labels = exp_versions
    y_lim = [[40, 75],[-0.3,0.8]]


    # Plotting
    c = load_my_colors()
    figure_size = (3, 2)
    font_size = 8
    font_name="Arial"  # width, height in inches
    axis_linewidth = 0.75
    line_width = 0.7
    color = c['light_gray']
    error_capsize = 5
    width = 0.5
    star_fontsize = 12
    ns_fontsize = 8

    # plot parameters with error bars
    n_parameters = len(parameter)
    for p in range(n_parameters):
        # make bar plot, which requires data in specific format:
        # x: number of datasets (exp_versions)
        fig, ax = custom_barplot(
                    x = exp_versions,
                    y = parameter[p],
                    y_ci = parameter_ci[p],
                    p_values = parameter_p_values[p],
                    y_reference_value = parameter_reference_value[p],
                    y_label = parameter_label[p],
                    y_ticks = y_ticks[p],
                    x_tick_labels = x_tick_labels,
                    y_lim = y_lim[p],
                    error_capsize = error_capsize,
                    color = color,
                    width = width,
                    star_fontsize = star_fontsize,
                    ns_fontsize = ns_fontsize,
                    font_name = font_name,
                    font_size = font_size,
                    axis_linewidth = axis_linewidth,
                    figure_size = figure_size,
                )
        # change spine width
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(0.5)
        # change tick width
        ax.tick_params(width=0.5)
        # save figure
        fig.tight_layout()
        fig.savefig(filenames[p])
        plt.close()


       
#%% ---------- plot_simple_regression_line_for_one_dataset ---------

def plot_simple_regression_line_for_one_dataset(
    merged_data: pd.DataFrame,
    independent_variable: str,
    filename: str
    ):

    data = merged_data.copy()
    # check that necessary variables are present and between 0 and 100
    data = _check_inputs(merged_data=data, independent_variable=independent_variable)

    # Fit linear model 
    formula = f"CFC_chose_highest_expected_value ~ {independent_variable}"
    model = ols(formula, data=data) # linear regression, no mixed effects
    result = model.fit()

    # New data grid (population-level predictions: random effect = 0 for unseen ID)
    x_min = data[independent_variable].min()
    x_max = data[independent_variable].max()
    x_grid = np.linspace(x_min, x_max, 100)

    new_data = pd.DataFrame({
        independent_variable: x_grid,
        "participant_ID": "new_subject"
    })

    # Confidence intervals for the mean prediction (fixed effects only)
    rhs = f"1 + {independent_variable}"
    prediction_df = _ols_mean_ci( 
        result,
        new_data,
        rhs_formula=rhs,
        alpha=0.05,
        mean_col="y_hat_CFC_chose_highest",
        lower_col="ci_low",
        upper_col="ci_high",
    )

    # ---- Plot styling ----
    c = load_my_colors()
    axis_linewidth = 0.75
    font_size = 8
    figure_size = (2, 2)
    font_name = "Arial"
    line_width = 0.7
    dot_size = 8
    dot_alpha = 0.38
    line_color = c["dark_gray"]
    y_ticks = [0, 25, 50, 75, 100]

    fig, ax = plt.subplots()
    fig.show()

    # reference lines
    ax.axhline(50, color="grey", linestyle="--", linewidth=0.7)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.7)
    # Regression li§ne
    ax.plot(
        prediction_df[independent_variable],
        prediction_df["y_hat_CFC_chose_highest"],
        color=line_color,
        linewidth=line_width,
        alpha=1,
    )
    # Confidence band
    ax.fill_between(
        prediction_df[independent_variable],
        prediction_df["ci_low"],
        prediction_df["ci_high"],
        alpha=0.4,
        facecolor=line_color,
    )
    # Scatter raw data (all points)
    ax.scatter(
        data[independent_variable],
        data["CFC_chose_highest_expected_value"],
        s=dot_size,
        marker="o",
        alpha=dot_alpha,
        edgecolors="None",
        facecolors="black",
        linewidths=line_width,
        zorder=3,
    )

    # Labels
    ax.set_xlabel(independent_variable, fontname=font_name, fontsize=font_size)
    ax.set_ylabel("CFC_chose_highest_expected_value", fontname=font_name, fontsize=font_size)

    # y-ticks
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(t) for t in y_ticks], fontfamily=font_name, fontsize=font_size)

    # ----- styling -----
    ax.tick_params(
        axis="both",
        which="major",
        length=4,
        width=axis_linewidth,
        direction="in",
        labelsize=font_size,
    )
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(axis_linewidth)

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname(font_name)
        lbl.set_fontsize(font_size)
  
    # ---- set limits explicitly - necessary to shade whole quadrants correctly ----
    xmin = np.nanmin(prediction_df[independent_variable])
    xmax = np.nanmax(prediction_df[independent_variable])
    margin = 0.05 * (xmax - xmin)
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim([0, 100]) # y axis limits
    #ax.set_xlim([-90, 90]) # x axis limits
    #ax.set_xlim([ax.get_xlim()[1], ax.get_xlim()[0]]) # set x axis limits with new max_x

    # shade the four quadrants
    _shade_quadrants(ax) 

    set_axes_size_inches(ax, target_width_in=figure_size[0], target_height_in=figure_size[1])

    # change spine width
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(0.5)
    # change tick width
    ax.tick_params(width=0.5)
    # save figure
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_simple_regression_parameters_for_one_dataset(
    merged_data: pd.DataFrame,
    independent_variable: str,
    filenames: list
    ):

    data = merged_data.copy()
    # check that necessary variables are present and between 0 and 100
    data = _check_inputs(merged_data=data, independent_variable=independent_variable)

    # linear regression
    formula = f"CFC_chose_highest_expected_value ~ {independent_variable}"

    # run regression on each dataset - here, we want "models" output which contains all model info from each regression
    (models, 
    intercepts, 
    intercepts_ci, 
    intercepts_p_values, 
    slopes, 
    slopes_ci, 
    slopes_p_values) = _run_regression_on_each_dataset(
                            merged_data=data,
                            independent_variable=independent_variable,
                            formula=formula,
                        )

    ''' 
    # prepare variables for plotting
    parameter_label = ['Intercept','Slope']
    parameter = [intercepts,slopes]
    parameter_ci = [intercepts_ci,slopes_ci]
    parameter_p_values = [intercepts_p_values,slopes_p_values]  
    parameter_reference_value = [50,0]
    y_ticks= [[50, 75, 100],(0,0.5,1)]
    x_tick_labels = ''
    y_lim = [[50, 100],[-0.2,1]]


    # Plotting
    c = load_my_colors()
    figure_size = (0.9, 0.7) # width, height in inches
    font_size = 8
    font_name="Arial"  # width, height in inches
    axis_linewidth = 0.75
    line_width = 0.7
    color = c['light_gray']
    error_capsize = 5
    width = 0.5
    star_fontsize = 12
    ns_fontsize = 8

    # plot parameters with error bars
    n_parameters = len(parameter)
    for p in range(n_parameters):
        # make bar plot, which requires data in specific format:
        # x: number of datasets (exp_versions)
        fig, ax = custom_barplot(
                    x = [1],
                    y = parameter[p],
                    y_ci = parameter_ci[p],
                    p_values = parameter_p_values[p],
                    y_reference_value = parameter_reference_value[p],
                    y_label = parameter_label[p],
                    y_ticks = y_ticks[p],
                    y_lim = y_lim[p],
                    x_tick_labels = x_tick_labels,
                    error_capsize = error_capsize,
                    color = color,
                    width = width,
                    star_fontsize = star_fontsize,
                    ns_fontsize = ns_fontsize,
                    font_name = font_name,
                    font_size = font_size,
                    axis_linewidth = axis_linewidth,
                    figure_size = figure_size,
                )
        # change spine width
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(0.5)
        # change tick width
        ax.tick_params(width=0.5)
        # save figure
        fig.tight_layout()
        fig.savefig(filenames[p])
        plt.close()

    '''

    # prepare variables for plotting
    plots = [
    dict(
        label="intercept",
        y=intercepts,
        ci=intercepts_ci,
        p=intercepts_p_values,
        ref=50,
        y_ticks=[50,75,100], 
        y_lim=[50,100]),
    dict(
        label="slope",
        y=slopes,
        ci=slopes_ci,
        p=slopes_p_values,
        ref=0,
        y_ticks=[0,0.5,1],    
        y_lim=[-0.1,1]),
    ]
   
    # Plotting
    c = load_my_colors()
    figure_size = (0.9, 0.7) # width, height in inches
    font_size = 8
    font_name="Arial"  # width, height in inches
    axis_linewidth = 0.75
    line_width = 0.7
    color = c['light_gray']
    error_capsize = 0
    width = 0.5
    star_fontsize = 10
    ns_fontsize = 8
    x_tick_labels = ['']


    for p, config in enumerate(plots):
        fig, ax = custom_barplot(
            x=['all'],
            y=config["y"],
            y_ci=config["ci"],
            p_values=config["p"],
            y_reference_value=config["ref"],
            y_label=config["label"],
            y_ticks=config["y_ticks"],
            x_tick_labels=x_tick_labels,
            y_lim=config["y_lim"],
            error_capsize=error_capsize,
            color=color,
            width=width,
            star_fontsize=star_fontsize,
            ns_fontsize=ns_fontsize,
            font_name=font_name,
            font_size=font_size,
            axis_linewidth=axis_linewidth,
            figure_size=figure_size,
        )
        # change spine width
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(0.5)
        # change tick width
        ax.tick_params(width=0.5)
        # save figure
        fig.tight_layout()
        fig.savefig(filenames[p])
        plt.close()

    # ------ also make pie chart of number of data points in each quadrant defined by x=0 and y=50 ------
    # get number of data points in each quadrant - make sure each data point is only counted once but all are counted
    right = data[independent_variable] > 0
    upper = data['CFC_chose_highest_expected_value'] > 50
    n_upper_right = (right & upper).sum()
    n_upper_left  = (~right & upper).sum()
    n_lower_left  = (~right & ~upper).sum()
    n_lower_right = (right & ~upper).sum()

    n_total = len(data)
    assert(n_total == n_upper_right + n_upper_left + n_lower_left + n_lower_right , "Error in counting data points in quadrants")
    # combine upper_right and lower_left into "rational"
    n_rational = n_upper_right + n_lower_left
    # calculate percentages
    perc_rational = (n_rational / n_total) * 100
    perc_upper_left = (n_upper_left / n_total) * 100
    perc_lower_right = (n_lower_right / n_total) * 100
    # prepare data for pie chart
    sizes = [perc_rational, 
             perc_upper_left, 
             perc_lower_right]
    colors = [c['light_blue'], 
              c['light_orange'], 
              c['light_brown']]
    # plot pie chart
    fig, ax = plt.subplots(figsize=(2,2))
    fig.show()
    wedges, texts, autotexts = ax.pie(
        sizes,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 7, 'fontname': font_name},
    )
    # equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    # set size
    set_axes_size_inches(ax, target_width_in=0.82, target_height_in=0.82)
    # change spine width
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(0.5)
    # change tick width
    ax.tick_params(width=0.5)
    # save figure
    fig.tight_layout()
    fig.savefig(filenames[2])
    plt.close()






#%% ---------- plot_simple_regression_line_for_all_datasets ---------
def plot_simple_regression_line_for_all_datasets(
    merged_data: pd.DataFrame,
    independent_variable: str,
    quadratic_regression: bool,
    filename: str
    ):

    # continue only if not quadratic 
    if not (quadratic_regression ):

        data = merged_data.copy()
        # check that necessary variables are present and between 0 and 100
        data = _check_inputs(merged_data=data, independent_variable=independent_variable)

        # set shape of model
        if quadratic_regression: # quadratic regression
            formula = f"CFC_chose_highest_expected_value ~ {independent_variable} + I({independent_variable}**2)"
        else: # linear regression
            formula = f"CFC_chose_highest_expected_value ~ {independent_variable}"

        # run regression on each dataset - here, we want "models" output which contains all model info from each regression
        (models, 
        intercepts, 
        intercepts_ci, 
        intercepts_p_values, 
        slopes, 
        slopes_ci, 
        slopes_p_values) = _run_regression_on_each_dataset(
                                merged_data=data,
                                independent_variable=independent_variable,
                                formula=formula,
                            )

        version_colors = _get_expID_colors(merged_data=data)
        exp_versions = data['exp_ID'].unique().tolist()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        # add lines for x = 0 and y = 50
        ax.axhline(50, color='black', linestyle='--', linewidth=1)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        # Loop through each experiment version
        for exp_version in exp_versions:
            version_color = version_colors[exp_version]
            model = models[exp_versions.index(exp_version)]

            # filter data for this exp_version
            if exp_version == 'all':
                df_version = data
            else:
                df_version = data[data['exp_ID'] == exp_version]

            # 1. Scatter plot of the data
            ax.scatter(df_version[independent_variable], 
                       df_version['CFC_chose_highest_expected_value'], 
                        color=version_color, 
                        alpha=0.3)
            
            # 2. Plot regression line
            # Make a smooth range of x values
            x_line = np.linspace(df_version[independent_variable].min(), df_version[independent_variable].max(), 100)
            # Use the fitted model to get predicted y for those x values
            y_line = model.predict(pd.DataFrame({independent_variable: x_line}))
            # Regression line
            ax.plot(x_line, 
                    y_line, 
                     color=version_color)

            # so far, can't plot confidence intervals for regression line easily

            # plot mean of x and y as error bars
            # calculate means
            mean_x = df_version[independent_variable].mean()
            mean_y = df_version['CFC_chose_highest_expected_value'].mean()
            # calculate 95% confidence intervals
            ci_x = stats.t.interval(0.95, len(df_version[independent_variable])-1, loc=mean_x, scale=stats.sem(df_version[independent_variable], nan_policy="omit"))
            ci_y = stats.t.interval(0.95, len(df_version['CFC_chose_highest_expected_value'])-1, loc=mean_y, scale=stats.sem(df_version['CFC_chose_highest_expected_value'], nan_policy="omit"))
            # plot error bars
            ax.errorbar(mean_x, 
                        mean_y, 
                        xerr=[[mean_x - ci_x[0]], [ci_x[1] - mean_x]], 
                        yerr=[[mean_y - ci_y[0]], [ci_y[1] - mean_y]], 
                        fmt='', 
                        color=version_color, 
                        ecolor=version_color, 
                        elinewidth=2.5, 
                        capsize=None,
                        label=exp_version)   

            ax.legend(fontsize=9)
            del df_version

        # figure settings
        ax.set_xlabel(f'{independent_variable} (%)', fontsize=14)
        ax.set_ylabel('pChose Gain in CFC (%)', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        # change spine width
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(0.5)
        # change tick width
        ax.tick_params(width=0.5)
        # save figure
        fig.tight_layout()
        fig.savefig(filename)
        plt.close()


#%% ---------- regression_full_formula_all_data ----------
def regression_full_formula_all_data(
    merged_data: pd.DataFrame,
    independent_variable: str,
    quadratic_regression: bool,
    filename: str
    ):

    data = merged_data.copy()

    # also run regression at level of entire dataset and print stats: 
    # independent variables = independent_variable descrived as input, click_desired, identify_best, LT_difficulty; dependent independent_variable = CFC_chose_highest_expected_value
    # include interactions
    # create formula
    formula = ""
    if quadratic_regression: # quadratic regression
        # formula = f'CFC_chose_highest_expected_value ~ {independent_variable} + I({independent_variable}**2) + C(click_desired) + C(identify_best) + C(LT_difficulty) + C(identify_best):C(click_desired) + C(LT_difficulty):C(identify_best) + C(click_desired):C(LT_difficulty) + C(click_desired):C(LT_difficulty):C(identify_best)'
        # throw error - quadratic regression too complex with all interactions
        raise ValueError("Quadratic regression with full formula and all interactions is not supported.")
    else: # linear regression
        formula = f'CFC_chose_highest_expected_value ~ {independent_variable} + C(click_desired) + C(identify_best) + C(identify_best):C(click_desired) + {independent_variable}:C(click_desired) + {independent_variable}:C(identify_best) + {independent_variable}:C(identify_best):C(click_desired) ',

    # use ols 
    model = ols(formula, data=data)
    result = model.fit()
    print(result.summary())

    # Save the summary to a text file
    with open(filename, 'w') as f:
        f.write(result.summary().as_text())



#%% ---------- plot_winning_regression_line_over_all_data ----------
def plot_winning_regression_line_over_all_data(
        merged_data: pd.DataFrame,
        independent_variable: str,
        filename: str
        ):
    
    data = merged_data.copy()
    # check that necessary variables are present and between 0 and 100
    data = _check_inputs(merged_data=data, independent_variable=independent_variable)

    # Fit mixed linear model
    formula = f'CFC_chose_highest_expected_value ~ {independent_variable} + C(identify_best)'
    model = ols(formula, data=data) # linear regression, no mixed effects
    result = model.fit()

    # New data for both identify_best = 0 and 1
    x_min = data[independent_variable].min()
    x_max = data[independent_variable].max()
    x_grid = np.linspace(x_min, x_max, 100)
    new_data = pd.DataFrame({
        independent_variable: np.tile(x_grid, 2),
        "identify_best": np.repeat([0, 1], repeats=len(x_grid)),
        "participant_ID": "new_subject" # unseen ID -> random effect = 0, gives population-level predictions AKA FIXED EFFECTS
    })

    # Compute confidence intervals for the *mean* prediction (fixed effects only) from a statsmodels OLS result.
    rhs = f"1 + {independent_variable} + C(identify_best)"   # Design matrix for fixed effects: intercept + x + C(identify_best)
    prediction_df = _ols_mean_ci( 
        result=result,
        new_data=new_data,
        rhs_formula=rhs,
        alpha=0.05,
        mean_col="y_hat_CFC_chose_highest",
        lower_col="ci_low",
        upper_col="ci_high"
    )

    # Plotting
    c = load_my_colors()
    axis_linewidth = 0.75
    font_size = 8
    figure_size = (2, 2)
    font_name="Arial"  # width, height in inches
    line_width = 0.7
    dot_size=8
    dot_alpha=0.38
    color_dict = {0: c['dark_gray'], 1: c['light_medium_gray']} #{0: c['medium_gray'], 1: c['dark_gray']}
    edge_color_dict = {0: "None", 1: "None"} #{0: c['medium_gray'], 1: "None"}
    face_color_dict = {0: "black", 1: c["medium_gray"]} #{0: "None", 1: c['dark_gray']}
    y_ticks=[0, 25, 50, 75, 100]
    

    fig, ax = plt.subplots()
    fig.show()    

    # add line for x = 0 and y = 50
    ax.axhline(50, color='grey', linestyle='--', linewidth=0.7)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.7)

    # Plot regression lines and confidence bands
    for grp, label in [(0, "identify_best = 0"), (1, "identify_best = 1")]:
        prediction_subset = prediction_df[prediction_df["identify_best"] == grp]

        # Regression line
        ax.plot(
            prediction_subset[independent_variable],
            prediction_subset["y_hat_CFC_chose_highest"],
            color=color_dict[grp],
            label=f"{label} (fit)",
            linewidth=line_width,
            alpha=1
        )

        # Confidence band
        ax.fill_between(
            prediction_subset[independent_variable],
            prediction_subset["ci_low"],
            prediction_subset["ci_high"],
            alpha=0.4,
            facecolor=color_dict[grp]
        )

        # scatter raw data
        ax.scatter(
            data.loc[data['identify_best'] == grp, independent_variable],
            data.loc[data['identify_best'] == grp, 'CFC_chose_highest_expected_value'],
            s=dot_size,
            marker='o',
            alpha=dot_alpha,
            edgecolors=edge_color_dict[grp],
            facecolors=face_color_dict[grp],
            linewidths=line_width,
            zorder=3,
        )
            

    ax.set_xlabel(independent_variable, fontname=font_name, fontsize=font_size)
    ax.set_ylabel("CFC_chose_highest_expected_value", fontname=font_name, fontsize=font_size)

    # y-ticks
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(
            [str(t) for t in y_ticks],
            fontfamily=font_name,
            fontsize=font_size,
        )
    # ----- styling -----
    # make tick marks turn inwards/outwards and set thickness
    ax.tick_params(
        axis="both",
        which="major",
        length=4,
        width=axis_linewidth,
        direction="in",
        labelsize=font_size,
    )
   # set axis lines connecting the tick marks
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(axis_linewidth)
    # impose font name and size for tick labels
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname(font_name)
        lbl.set_fontsize(font_size)

    # change spine width
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(0.5)
    # change tick width
    ax.tick_params(width=0.5)

    # ---- set limits explicitly - necessary to shade whole quadrants correctly ----
    xmin = np.nanmin(prediction_df[independent_variable])
    xmax = np.nanmax(prediction_df[independent_variable])
    margin = 0.05 * (xmax - xmin)
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim([0, 100]) # y axis limits
    #ax.set_xlim([-90, 90]) # x axis limits
    #ax.set_xlim([ax.get_xlim()[1], ax.get_xlim()[0]]) # set x axis limits with new max_x

    # shade the four quadrants
    _shade_quadrants(ax) 
    # set height of axes
    set_axes_size_inches(ax, target_width_in=figure_size[0], target_height_in=figure_size[1])
    # change spine width
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(0.5)
    # change tick width
    ax.tick_params(width=0.5)
    # save figure
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()






def plot_winning_regression_line_with_difficulty_over_all_data(
        merged_data: pd.DataFrame,
        independent_variable: str,
        filename: str
    ):

    data = merged_data.copy()
    # check that necessary variables are present and between 0 and 100
    data = _check_inputs(merged_data=data, independent_variable=independent_variable)
    
    # --- Fit mixed linear model with extra categorical (no interactions) ---
    formula = f"CFC_chose_highest_expected_value ~ {independent_variable} + C(identify_best) + C(LT_unequal_difficulty_binary)"
    model = ols(formula, data=data) # linear regression, no mixed effects
    result = model.fit()

    # --- New data grid: x over both binaries (identify_best × LT) ---
    x_min = data[independent_variable].min()
    x_max = data[independent_variable].max()
    x_grid = np.linspace(x_min, x_max, 100)

    new_data = pd.DataFrame({
        independent_variable: np.tile(x_grid, 4),
        "identify_best": np.repeat([0, 1, 0, 1], repeats=len(x_grid)),
        "LT_unequal_difficulty_binary": np.repeat([0, 0, 1, 1], repeats=len(x_grid)),
        # unseen ID -> RE=0 -> population-level (fixed effects) predictions
        "participant_ID": "new_subject"
    })

   # --- Fixed-effects mean + CI (no interactions in RHS) ---
    rhs = f"1 + {independent_variable} + C(identify_best) + C(LT_unequal_difficulty_binary)" 
    prediction_df = _ols_mean_ci( 
        result=result,
        new_data=new_data,
        rhs_formula=rhs,
        alpha=0.05,
        mean_col="y_hat_CFC_chose_highest",
        lower_col="ci_low",
        upper_col="ci_high"
    )
    

    # --- Plot: all four lines on ONE plot ---
    c = load_my_colors()
    axis_linewidth = 0.75
    font_size = 8
    figure_size = (2.8, 2.4)
    font_name = "Arial"
    line_width = 0.7
    dot_size = 18.0
    dot_alpha = 0.3
    y_ticks = [0, 25, 50, 75, 100]

    # color by LT, linestyle by identify_best
    lt_color = {0: c['medium_gray'], 1: c['dark_gray']}
    ib_style = {0: '-', 1: '--'}

    fig, ax = plt.subplots(figsize=figure_size)

    # reference lines
    ax.axhline(50, color='grey', linestyle='--', linewidth=1)
    ax.axvline(0,  color='grey', linestyle='--', linewidth=1)

    # loop over all 4 combos
    for lt in [0, 1]:
        for ib in [0, 1]:
            subset = prediction_df[
                (prediction_df["LT_unequal_difficulty_binary"] == lt) &
                (prediction_df["identify_best"] == ib)
            ]

            label = f"LT={lt}, identify_best={ib}"

            # regression line
            ax.plot(
                subset[independent_variable],
                subset["y_hat_CFC_chose_highest"],
                color=lt_color[lt],
                linestyle=ib_style[ib],
                linewidth=line_width,
                label=label,
                alpha=1
            )

            # CI band (same color; transparency distinguishes it)
            ax.fill_between(
                subset[independent_variable],
                subset["ci_low"],
                subset["ci_high"],
                facecolor=lt_color[lt],
                alpha=0.18
            )

            # raw points for this combo
            raw = data[
                (data["LT_unequal_difficulty_binary"] == lt) &
                (data["identify_best"] == ib)
            ]
            ax.scatter(
                raw[independent_variable],
                raw["CFC_chose_highest_expected_value"],
                s=dot_size,
                marker='o',
                alpha=dot_alpha,
                facecolor=lt_color[lt],
                linewidths=0,
                zorder=3
            )

    ax.set_xlabel(independent_variable, fontname=font_name, fontsize=font_size)
    ax.set_ylabel("CFC_chose_highest_expected_value", fontname=font_name, fontsize=font_size)

    # y-ticks
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(
            [str(t) for t in y_ticks],
            fontfamily=font_name,
            fontsize=font_size,
        )
    # ----- styling -----
    # make tick marks turn inwards/outwards and set thickness
    ax.tick_params(
        axis="both",
        which="major",
        length=4,
        width=axis_linewidth,
        direction="in",
        labelsize=font_size,
    )
   # set axis lines connecting the tick marks
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(axis_linewidth)
    # impose font name and size for tick labels
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname(font_name)
        lbl.set_fontsize(font_size)

    # ---- set limits explicitly - necessary to shade whole quadrants correctly ----
    xmin = np.nanmin(prediction_df[independent_variable])
    xmax = np.nanmax(prediction_df[independent_variable])
    margin = 0.05 * (xmax - xmin)
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim([0, 100]) # y axis limits
    #ax.set_xlim([-90, 90]) # x axis limits
    #ax.set_xlim([ax.get_xlim()[1], ax.get_xlim()[0]]) # set x axis limits with new max_x

    # shade the four quadrants
    _shade_quadrants(ax) 

    # set height of axes
    set_axes_size_inches(ax, target_width_in=figure_size[0], target_height_in=figure_size[1])
    # change spine width
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(0.5)
    # change tick width
    ax.tick_params(width=0.5)
    # save figure
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()



#%% ---------- plot_winning_regression_parameters_all_data ----------

def plot_winning_regression_parameters_all_data(
    merged_data: pd.DataFrame,
    independent_variable: str,
    filenames: list,
    ):

    data = merged_data.copy()




    ''' 
    df_copy = df.copy()
    #-------- prep data --------
    # for each df, average per participant per valence 
    LT_mean_per_participant_per_valence = df_copy['LearningTask'].groupby(['participant_ID', 'exp_ID', 'is_gain_trial'], as_index=False)['correct'].mean()
    SC_mean_per_participant_per_valence = df_copy['SymbolChoice'].groupby(['participant_ID', 'exp_ID', 'is_gain_trial', 'identify_best'], as_index=False)['correct'].mean()
    CFC_mean_per_participant = df_copy['PairChoice'].groupby(['participant_ID', 'exp_ID', 'identify_best'], as_index=False)['chose_highest_expected_value'].mean()
    # scale to percentage
    LT_mean_per_participant_per_valence['correct'] = LT_mean_per_participant_per_valence['correct'] * 100 
    SC_mean_per_participant_per_valence['correct'] = SC_mean_per_participant_per_valence['correct'] * 100 
    CFC_mean_per_participant['chose_highest_expected_value'] = CFC_mean_per_participant['chose_highest_expected_value'] * 100 
    # create wide data frame to have one column gain_correct and loss_correct for LT and SC 
    LT_mean_per_participant_per_valence_wide = LT_mean_per_participant_per_valence.pivot(index=['participant_ID'], columns='is_gain_trial', values='correct').reset_index()
    LT_mean_per_participant_per_valence_wide.columns = ['participant_ID', 'LT_correct_loss', 'LT_correct_gain']
    SC_mean_per_participant_per_valence_wide = SC_mean_per_participant_per_valence.pivot(index=['participant_ID','identify_best'], columns='is_gain_trial', values='correct').reset_index()
    SC_mean_per_participant_per_valence_wide.columns = ['participant_ID', 'identify_best', 'SC_correct_loss', 'SC_correct_gain']
    # subtract loss from gain to get single performance measure per participant for LT and SC
    LT_mean_per_participant_per_valence_wide['LT_correct_gain_minus_loss'] = LT_mean_per_participant_per_valence_wide['LT_correct_gain'] - LT_mean_per_participant_per_valence_wide['LT_correct_loss']
    SC_mean_per_participant_per_valence_wide['SC_correct_gain_minus_loss'] = SC_mean_per_participant_per_valence_wide['SC_correct_gain'] - SC_mean_per_participant_per_valence_wide['SC_correct_loss']
    # rename columns to be clear where it came from
    CFC_mean_per_participant.rename(columns={'chose_highest_expected_value': 'CFC_chose_highest_expected_value'}, inplace=True)
    # merge dfs to get one df with LT, SC, and CFC performance per participant per valence
    merged_data = pd.merge(CFC_mean_per_participant, SC_mean_per_participant_per_valence_wide, on=['participant_ID', 'identify_best'])
    merged_data = pd.merge(merged_data, LT_mean_per_participant_per_valence_wide, on=['participant_ID'])

    versions_to_include_1 = ['cd1_2025_click_desired_0_identify_best_0',
                            'cd1_2025_click_desired_0_identify_best_1',
                            'cd1_2025_click_desired_1_identify_best_0',
                            'cd1_2025_click_desired_1_identify_best_1']
    merged_data = merged_data[merged_data['exp_ID'].isin(versions_to_include_1)]




    df1 = merged_data
    df2 = data

    # df1, df2 are your dataframes
    id_col = "participant_ID"

    # Use participant_ID as index so rows align by participant_ID
    a = df1.set_index(id_col)
    b = df2.set_index(id_col)

    # Compare only columns that exist in both dataframes (excluding the ID)
    common_cols = a.columns.intersection(b.columns)
    a = a[common_cols]
    b = b[common_cols]

    # Treat NaN == NaN as equal
    diff_mask = a.ne(b) & ~(a.isna() & b.isna())

    # check if there are same participants (participant_IDs) in both dataframes, print unmatched participant_IDs if not
    if not a.index.equals(b.index):
        unmatched_a = a.index.difference(b.index)
        unmatched_b = b.index.difference(a.index)        
        print(f"\nWarning: participant_IDs do not match between dataframes. Total in df1: {len(a.index)}, Total in df2: {len(b.index)}, Unmatched in df1: {len(unmatched_a)}, Unmatched in df2: {len(unmatched_b)}")

    '''




    data = _check_inputs(
        merged_data=data,
        independent_variable=independent_variable,
        )
    text_filename = filenames[0]
    plot_filenames = filenames[1:]

    formula = f'CFC_chose_highest_expected_value ~ {independent_variable} + C(identify_best)'
    
    # use ols_model
    model = ols(formula, data=data).fit()
    print(model.summary())

    # Save the summary to a text file
    with open(text_filename, 'w') as f:
        f.write(model.summary().as_text())


    # ------ plotting of parameters ------    

    # extract slope with CI, p-values
    slope = model.params[independent_variable]
    slope_ci = model.conf_int().loc[independent_variable].to_list()
    slope_p_value = model.pvalues[independent_variable]
    # extract intercept wit CI and pvalue
    intercept = model.params['Intercept']
    intercept_ci = model.conf_int().loc['Intercept'].to_list()
    intercept_p_value = model.pvalues['Intercept']
    # extract coef for identify_best = 1 with CI and pvalue
    coef_identify_best = model.params['C(identify_best)[T.1]']
    coef_identify_best_ci = model.conf_int().loc['C(identify_best)[T.1]'].to_list()
    coef_identify_best_p_value = model.pvalues['C(identify_best)[T.1]']
    # compute intercept for identify_best = 1
    intercept_for_identify_best_1 = coef_identify_best + intercept
    intercept_for_identify_best_1_ci = coef_identify_best_ci + intercept # shift both bounds by intercept

    '''  
    # compute CI specifically for intercept_for_identify_best_1 which is combination of two coefs
    # formula: square root (var(intercept) + var(coef_identify_best) + 2*cov(intercept, coef_identify_best))
    var_intercept = model.cov_params().loc['Intercept','Intercept']
    var_coef_identify_best = model.cov_params().loc['C(identify_best)[T.1]','C(identify_best)[T.1]']
    cov_intercept_coef_identify_best = model.cov_params().loc['Intercept','C(identify_best)[T.1]']
    intercept_for_identify_best_1_se = np.sqrt(var_intercept + var_coef_identify_best + 2*cov_intercept_coef_identify_best)
    z = norm.ppf(1 - 0.05 / 2)  # 95% CI
    intercept_for_identify_best_1_ci = [
        intercept_for_identify_best_1 - z * intercept_for_identify_best_1_se,
        intercept_for_identify_best_1 + z * intercept_for_identify_best_1_se
    ]   
    '''

    # prepare variables for plotting
    plots = [
    dict(label="intercept",       
         y=intercept,                   
         ci=intercept_ci,       
         p=intercept_p_value,       
         ref=50, 
         y_ticks=[50,75,100], 
         y_lim=[50,100]),
    dict(label="slope",           
         y=slope,                       
         ci=slope_ci,           
         p=slope_p_value,           
         ref=0,  
         y_ticks=[0,0.5,1],    
         y_lim=[-0.1,1]),
    dict(label="CFC+",  
         y=intercept_for_identify_best_1, 
         ci=intercept_for_identify_best_1_ci, 
         p=coef_identify_best_p_value, 
         ref=50, 
         y_ticks=[50,75,100], 
         y_lim=[50,100]),
    ]
    # Plotting
    c = load_my_colors()
    figure_size = (0.9, 0.7) # width, height in inches
    font_size = 8
    font_name="Arial"  # width, height in inches
    axis_linewidth = 0.75
    line_width = 0.7
    color = c['light_gray']
    error_capsize = 0
    width = 0.5
    star_fontsize = 10
    ns_fontsize = 8
    x_tick_labels = ['']
    exp_versions = ['all']

    for p, config in enumerate(plots):
        fig, ax = custom_barplot(
            x=exp_versions,
            y=[config["y"]],
            y_ci=[config["ci"]],
            p_values=[config["p"]],
            y_reference_value=config["ref"],
            y_label=config["label"],
            y_ticks=config["y_ticks"],
            y_lim=config["y_lim"],
            x_tick_labels=x_tick_labels,
            error_capsize=error_capsize,
            color=color,
            width=width,
            star_fontsize=star_fontsize,
            ns_fontsize=ns_fontsize,
            font_name=font_name,
            font_size=font_size,
            axis_linewidth=axis_linewidth,
            figure_size=figure_size,
        )
        # change spine width
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(0.5)
        # change tick width
        ax.tick_params(width=0.5)
        # save figure
        fig.tight_layout()
        fig.savefig(plot_filenames[p])
        plt.close()

    # ------ also make stacked bar plot of intercepts (identify_best = 0 and 1) ------
    fig, ax = plt.subplots()
    large_figure_size =[1.15, 0.7]
    fig,ax = _plot_intercept_stacked_bar(
        fig,
        ax,
        intercept=intercept,
        intercept_ci=intercept_ci,
        intercept_p_value=intercept_p_value,
        coef_identify_best=coef_identify_best,
        coef_identify_best_ci=coef_identify_best_ci,
        coef_identify_best_p_value=coef_identify_best_p_value,
        width=width*1.2,
        color=c,
        star_fontsize=star_fontsize,
        figure_size = large_figure_size,
    )
    # change spine width
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(0.5)
    # change tick width
    ax.tick_params(width=0.5)
    # save figure
    fig.tight_layout()
    fig.savefig(plot_filenames[3])
    plt.close()
    

    # ------ make pie chart ------

     # get number of data points in each quadrant - make sure each data point is only counted once but all are counted
    right = data[independent_variable] > 0
    upper = data['CFC_chose_highest_expected_value'] > 50
    n_upper_right = (right & upper).sum()
    n_upper_left  = (~right & upper).sum()
    n_lower_left  = (~right & ~upper).sum()
    n_lower_right = (right & ~upper).sum()

    n_total = len(data)
    assert(n_total == n_upper_right + n_upper_left + n_lower_left + n_lower_right , "Error in counting data points in quadrants")
    # combine upper_right and lower_left into "rational"
    n_rational = n_upper_right + n_lower_left
    # calculate percentages
    perc_rational = (n_rational / n_total) * 100
    perc_upper_left = (n_upper_left / n_total) * 100
    perc_lower_right = (n_lower_right / n_total) * 100
    # prepare data for pie chart
    sizes = [perc_rational, 
             perc_upper_left, 
             perc_lower_right]
    colors = [c['light_blue'], 
              c['light_orange'], 
              c['light_brown']]
    # plot pie chart
    fig, ax = plt.subplots()
    fig.show()
    wedges, texts, autotexts = ax.pie(
        sizes,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 7, 'fontname': font_name},
    )
    # equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    # change spine width
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(0.5)
    # change tick width
    ax.tick_params(width=0.5)
    # set size
    set_axes_size_inches(ax, target_width_in=0.82, target_height_in=0.82)
    # save figure
    fig.tight_layout()
    fig.savefig(plot_filenames[4])
    plt.close()


def _shade_quadrants(ax, x_split=0, y_split=50, alpha=1.0, zorder=0):
    # Make sure limits reflect everything already plotted
    ax.relim()
    ax.autoscale_view()

    c = load_my_colors()
    quadrant_colors = {
        "upper_left": c["light_orange"],
        "lower_right": c["light_brown"],
        "upper_right": c["light_blue"],
        "lower_left": c["light_blue"],
    }

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    if y_split < y_max:
        y0 = max(y_split, y_min)
        ax.fill_betweenx([y0, y_max], x_min, x_split,
                         facecolor=quadrant_colors["upper_left"], alpha=alpha, zorder=zorder)
        ax.fill_betweenx([y0, y_max], x_split, x_max,
                         facecolor=quadrant_colors["upper_right"], alpha=alpha, zorder=zorder)

    if y_min < y_split:
        y1 = min(y_split, y_max)
        ax.fill_betweenx([y_min, y1], x_min, x_split,
                         facecolor=quadrant_colors["lower_left"], alpha=alpha, zorder=zorder)
        ax.fill_betweenx([y_min, y1], x_split, x_max,
                         facecolor=quadrant_colors["lower_right"], alpha=alpha, zorder=zorder)


def _plot_intercept_stacked_bar(
    fig,
    ax,
    intercept,
    intercept_ci,
    intercept_p_value,
    coef_identify_best,
    coef_identify_best_ci,
    coef_identify_best_p_value,
    color,
    font_size=8,
    font_name="Arial",
    axis_linewidth=0.75,
    width=0.5,
    error_capsize=0,
    star_fontsize=12,
    ns_fontsize=8,
    figure_size=[1.2, 2.5]
):

    # Normalize inputs (accept scalars / 1D CI lists)
    intercept = float(np.asarray(intercept).squeeze())
    coef_identify_best = float(np.asarray(coef_identify_best).squeeze())
    intercept_ci = np.asarray(intercept_ci).squeeze()
    coef_identify_best_ci = np.asarray(coef_identify_best_ci).squeeze()

    c = color

    fig.show()

    x_positions = np.array([0.1, 0.9]) # [0, 1] NB: x axis length set to: -0.5, 1.5

    # x=1: intercept only
    ax.bar(
        x_positions[0],
        intercept,
        color=c['dark_gray'],
        edgecolor=c['dark_gray'],
        width=width,
        zorder=3,
    )

    # x=2: stacked intercept + identify_best effect
    ax.bar(
        x_positions[1],
        intercept,
        color=c['dark_gray'],
        edgecolor=c['dark_gray'],
        width=width,
        zorder=3,
    )
    ax.bar(
        x_positions[1],
        coef_identify_best,
        bottom=intercept, # allows stacking
        color=c['light_medium_gray'],
        edgecolor=c['light_medium_gray'],
        width=width,
        zorder=3,
    )

    # add error bars - seperate for intercept and for intercept + identify_best
    # transform CI to yerr format
    yerr1 = intercept_ci[1] - intercept # single value to be added and subtracted
    ax.errorbar(
        x_positions[0],
        intercept,
        yerr=yerr1,
        fmt="k.",
        ecolor="black",
        elinewidth=axis_linewidth,
        capsize=error_capsize,
        zorder=4,
    )

    # Error bar for stacked total (x=2): intercept + coef_identify_best
    stacked_total = intercept + coef_identify_best
    # transform CI to yerr format
    yerr2 = coef_identify_best_ci[1] - coef_identify_best # single value to be added and subtracted
    ax.errorbar(
        x_positions[1],
        stacked_total,
        yerr=yerr2,
        fmt="k.",
        ecolor="black",
        elinewidth=axis_linewidth,
        capsize=error_capsize,
        zorder=4,
    )

    # reference line
    ax.axhline(50, color='gray', linestyle='--', lw=1)

    # add stars over first bar
    star_text_1 = significance_stars(intercept_p_value)
    ax.text(
        x=x_positions[0],
        y=(intercept + yerr1 - 5),
        s=star_text_1,
        fontsize=star_fontsize,
        fontname=font_name,
        ha='center',
        va='bottom',
        zorder=4,
    )
    # add horizontal bar above the two bars (middle)
    bar_y = intercept + coef_identify_best + yerr2 + 10
    bar_x1 = x_positions[0] #- width / 2
    bar_x2 = x_positions[1] #+ width / 2
    ax.plot(
        [bar_x1, bar_x2],
        [bar_y, bar_y],
        color='black',
        lw=0.75,
        zorder=4,
    )
    # add stars over second bar
    star_text_2 = significance_stars(coef_identify_best_p_value)
    ax.text(
        x=(x_positions[0] + x_positions[1]) / 2,
        y=bar_y - 5,
        s=star_text_2,    
        fontsize=star_fontsize,
        fontname=font_name,
        ha='center',
        va='bottom',
        zorder=4,
    )
    # axes
    ax.set_ylabel('intercept', fontsize=font_size, fontname=font_name)
    ax.set_xlim(-0.5, 1.5) # ax.set_xlim(x_lim[0], len(x)-0.5)
    ax.set_xticks([])  #ax.set_xticks(x_positions)
    #ax.set_xticklabels(['CFC-', 'CFC+'], fontsize=6) #rotation=45, ha='right'
    ax.set_ylim(50, 100)
    ax.set_yticks([50, 75, 100])
    ax.set_yticklabels(['50', '75', '100'], fontfamily=font_name, fontsize=font_size)

    ax.tick_params(
        axis="both",
        which="major",
        length=4,
        width=axis_linewidth,
        direction="in",
        labelsize=font_size,
    )
    ax.grid(False)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    for spine in ("left",):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
    for axis in ['top','left','right']:
        ax.spines[axis].set_linewidth(axis_linewidth)
    ax.xaxis.set_ticks_position('none')
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname(font_name)
        lbl.set_fontsize(font_size)


    set_axes_size_inches(ax, target_width_in=figure_size[0], target_height_in=figure_size[1])

    return fig, ax



#%% plot_logodds_of_DV_in_winning_regression_line_over_all_data
def plot_logodds_of_DV_in_winning_regression_line_over_all_data(
                                merged_data: pd.DataFrame,
                                independent_variable: str,
                                filename: str
                            ):
    '''    
    Plots the winning regression line (with confidence intervals) for the relationship between an independent variable and CFC_chose_highest_expected_value, 
    separately for identify_best = 0 and 1. Also plots raw data points and shades quadrants.

    Parameters:
    - merged_data: DataFrame containing the data, must include columns for 
        - participant_ID, 
        - identify_best, 
        - CFC_chose_highest_expected_value, 
            - !!! CRUCIALLY this variable is provided in log odds form (not percentage) to be compatible with the logit regression model.
        - specified independent_variable.
    - independent_variable: String name of the independent variable to plot on the x-axis.
    - filename: String path to save the resulting plot (e.g., "regression_plot.png").


    ''' 

    data = merged_data.copy()
    # check that necessary variables are present and between 0 and 100
    data = _check_inputs(merged_data=data, independent_variable=independent_variable)

    # Fit mixed linear model
    formula = f'CFC_chose_highest_expected_value ~ {independent_variable} + C(identify_best)'
    model = ols(formula, data=data) # linear regression, no mixed effects
    result = model.fit()

    # New data for both identify_best = 0 and 1
    x_min = data[independent_variable].min()
    x_max = data[independent_variable].max()
    x_grid = np.linspace(x_min, x_max, 100)
    new_data = pd.DataFrame({
        independent_variable: np.tile(x_grid, 2),
        "identify_best": np.repeat([0, 1], repeats=len(x_grid)),
        "participant_ID": "new_subject" # unseen ID -> random effect = 0, gives population-level predictions AKA FIXED EFFECTS
    })

    # Compute confidence intervals for the *mean* prediction (fixed effects only) from a statsmodels OLS result.
    rhs = f"1 + {independent_variable} + C(identify_best)"   # Design matrix for fixed effects: intercept + x + C(identify_best)
    prediction_df = _ols_mean_ci( 
        result=result,
        new_data=new_data,
        rhs_formula=rhs,
        alpha=0.05,
        mean_col="y_hat_CFC_chose_highest",
        lower_col="ci_low",
        upper_col="ci_high"
    )

    # Plotting
    c = load_my_colors()
    axis_linewidth = 0.75
    font_size = 8
    figure_size = (2, 2)
    font_name="Arial"  # width, height in inches
    line_width = 0.7
    dot_size=8
    dot_alpha=0.38
    color_dict = {0: c['dark_gray'], 1: c['light_medium_gray']} #{0: c['medium_gray'], 1: c['dark_gray']}
    edge_color_dict = {0: "None", 1: "None"} #{0: c['medium_gray'], 1: "None"}
    face_color_dict = {0: "black", 1: c["medium_gray"]} #{0: "None", 1: c['dark_gray']}
    #y_ticks=[0, 25, 50, 75, 100]
    

    fig, ax = plt.subplots()
    fig.show()    

    # add line for x = 0 and y = 50
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.7)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.7)

    # Plot regression lines and confidence bands
    for grp, label in [(0, "identify_best = 0"), (1, "identify_best = 1")]:
        prediction_subset = prediction_df[prediction_df["identify_best"] == grp]

        # Regression line
        ax.plot(
            prediction_subset[independent_variable],
            prediction_subset["y_hat_CFC_chose_highest"],
            color=color_dict[grp],
            label=f"{label} (fit)",
            linewidth=line_width,
            alpha=1
        )

        # Confidence band
        ax.fill_between(
            prediction_subset[independent_variable],
            prediction_subset["ci_low"],
            prediction_subset["ci_high"],
            alpha=0.4,
            facecolor=color_dict[grp]
        )

        # scatter raw data
        ax.scatter(
            data.loc[data['identify_best'] == grp, independent_variable],
            data.loc[data['identify_best'] == grp, 'CFC_chose_highest_expected_value'],
            s=dot_size,
            marker='o',
            alpha=dot_alpha,
            edgecolors=edge_color_dict[grp],
            facecolors=face_color_dict[grp],
            linewidths=line_width,
            zorder=3,
        )
            

    ax.set_xlabel(independent_variable, fontname=font_name, fontsize=font_size)
    ax.set_ylabel("CFC_chose_highest_expected_value", fontname=font_name, fontsize=font_size)

    # ----- styling -----
    # make tick marks turn inwards/outwards and set thickness
    ax.tick_params(
        axis="both",
        which="major",
        length=4,
        width=axis_linewidth,
        direction="in",
        labelsize=font_size,
    )
   # set axis lines connecting the tick marks
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(axis_linewidth)
    # impose font name and size for tick labels
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname(font_name)
        lbl.set_fontsize(font_size)

    # change spine width
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(0.5)
    # change tick width
    ax.tick_params(width=0.5)

    # ---- set limits explicitly - necessary to shade whole quadrants correctly ----
    xmin = np.nanmin(prediction_df[independent_variable])
    xmax = np.nanmax(prediction_df[independent_variable])
    margin = 0.05 * (xmax - xmin)
    ax.set_xlim(xmin - margin, xmax + margin)

    # set height of axes
    set_axes_size_inches(ax, target_width_in=figure_size[0], target_height_in=figure_size[1])
    # change spine width
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(0.5)
    # change tick width
    ax.tick_params(width=0.5)
    # save figure
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()




def plot_logit_winning_regression_parameters_all_data(
    merged_data: pd.DataFrame,
    independent_variable: str,
    filenames: list
    ):

    data = merged_data.copy()
    data = _check_inputs(
        merged_data=data,
        independent_variable=independent_variable,
        )
    text_filename = filenames[0]
    plot_filenames = filenames[1:]

    formula = f'CFC_chose_highest_expected_value ~ {independent_variable} + C(identify_best)'
    
    # use ols_model
    model = ols(formula, data=data).fit()
    print(model.summary())

    # Save the summary to a text file
    with open(text_filename, 'w') as f:
        f.write(model.summary().as_text())


    # ------ plotting of parameters ------    

    # extract slope with CI, p-values
    slope = model.params[independent_variable]
    slope_ci = model.conf_int().loc[independent_variable].to_list()
    slope_p_value = model.pvalues[independent_variable]
    # extract intercept wit CI and pvalue
    intercept = model.params['Intercept']
    intercept_ci = model.conf_int().loc['Intercept'].to_list()
    intercept_p_value = model.pvalues['Intercept']
    # extract coef for identify_best = 1 with CI and pvalue
    coef_identify_best = model.params['C(identify_best)[T.1]']
    coef_identify_best_ci = model.conf_int().loc['C(identify_best)[T.1]'].to_list()
    coef_identify_best_p_value = model.pvalues['C(identify_best)[T.1]']
    # compute intercept for identify_best = 1
    intercept_for_identify_best_1 = coef_identify_best + intercept
    intercept_for_identify_best_1_ci = coef_identify_best_ci + intercept # shift both bounds by intercept

    # prepare variables for plotting
    plots = [
    dict(label="intercept",       
         y=intercept,                   
         ci=intercept_ci,       
         p=intercept_p_value,       
         ref=0, 
         y_ticks=[-5,0,5], 
         y_lim=[-5,5]),
    dict(label="slope",           
         y=slope,                       
         ci=slope_ci,           
         p=slope_p_value,           
         ref=0,  
         y_ticks=[0,0.5,1],    
         y_lim=[-0.1,1]),
    dict(label="CFC+",  
         y=intercept_for_identify_best_1, 
         ci=intercept_for_identify_best_1_ci, 
         p=coef_identify_best_p_value, 
         ref=0, 
         y_ticks=[-5,0,5], 
         y_lim=[-5,5]),
    ]
    # Plotting
    c = load_my_colors()
    figure_size = (0.9, 0.7) # width, height in inches
    font_size = 8
    font_name="Arial"  # width, height in inches
    axis_linewidth = 0.75
    line_width = 0.7
    color = c['light_gray']
    error_capsize = 0
    width = 0.5
    star_fontsize = 10
    ns_fontsize = 8
    x_tick_labels = ['']
    exp_versions = ['all']

    for p, config in enumerate(plots):
        fig, ax = custom_barplot(
            x=exp_versions,
            y=[config["y"]],
            y_ci=[config["ci"]],
            p_values=[config["p"]],
            y_reference_value=config["ref"],
            y_label=config["label"],
            y_ticks=config["y_ticks"],
            y_lim=config["y_lim"],
            x_tick_labels=x_tick_labels,
            error_capsize=error_capsize,
            color=color,
            width=width,
            star_fontsize=star_fontsize,
            ns_fontsize=ns_fontsize,
            font_name=font_name,
            font_size=font_size,
            axis_linewidth=axis_linewidth,
            figure_size=figure_size,
        )
        # change spine width
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(0.5)
        # change tick width
        ax.tick_params(width=0.5)
        # save figure
        fig.tight_layout()
        fig.savefig(plot_filenames[p])
        plt.close()

    # ------ make pie chart ------

     # get number of data points in each quadrant - make sure each data point is only counted once but all are counted
    right = data[independent_variable] > 0
    upper = data['CFC_chose_highest_expected_value'] > 0
    n_upper_right = (right & upper).sum()
    n_upper_left  = (~right & upper).sum()
    n_lower_left  = (~right & ~upper).sum()
    n_lower_right = (right & ~upper).sum()

    n_total = len(data)
    assert(n_total == n_upper_right + n_upper_left + n_lower_left + n_lower_right , "Error in counting data points in quadrants")
    # combine upper_right and lower_left into "rational"
    n_rational = n_upper_right + n_lower_left
    # calculate percentages
    perc_rational = (n_rational / n_total) * 100
    perc_upper_left = (n_upper_left / n_total) * 100
    perc_lower_right = (n_lower_right / n_total) * 100
    # prepare data for pie chart
    sizes = [perc_rational, 
             perc_upper_left, 
             perc_lower_right]
    colors = [c['light_blue'], 
              c['light_orange'], 
              c['light_brown']]
    # plot pie chart
    fig, ax = plt.subplots()
    fig.show()
    wedges, texts, autotexts = ax.pie(
        sizes,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 7, 'fontname': font_name},
    )
    # equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    # change spine width
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(0.5)
    # change tick width
    ax.tick_params(width=0.5)
    # set size
    set_axes_size_inches(ax, target_width_in=0.82, target_height_in=0.82)
    # save figure
    fig.tight_layout()
    fig.savefig(plot_filenames[4])
    plt.close()

