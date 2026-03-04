from scipy import stats
import statsmodels.formula.api as smf
import numpy as np
import os
import pandas as pd
from Functions.text_p_value import text_p_value


# =============================================================================
# Helper functions (stats + printing)
# =============================================================================

def run_one_sample_ttest_and_format(sample_values, population_mean_under_null: float, label: str) -> None:
    """
    One-sample t-test against a known mean, printing:
    mean ± 95% CI half-width, t(df), p.
    Assumes df = n - 1 (your request).
    """

    sample_values = sample_values.dropna()
    n = int(sample_values.shape[0])
    df = n - 1

    ttest_result = stats.ttest_1samp(sample_values, population_mean_under_null)
    p_value_text = text_p_value(ttest_result.pvalue)

    mean = sample_values.mean()
    sem = stats.sem(sample_values)  # standard error of the mean
    ci_low, ci_high = stats.t.interval(0.95, df=df, loc=mean, scale=sem)
    ci_half_width = (ci_high - ci_low) / 2

    print(
        f"\n{label}: "
        f"mean={mean:.2f}±{ci_half_width:.2f}, "
        f"t({df})={ttest_result.statistic:.2f}, "
        f"{p_value_text};"
    )


def run_paired_ttest_and_format(first_condition_values, second_condition_values, label: str) -> float:
    """
    Paired t-test (first - second), printing:
    mean difference ± 95% CI half-width, t(df), p.
    Assumes df = n - 1 (your request).
    Returns paired-samples Cohen's d (mean(diff)/sd(diff), ddof=1).
    """
    paired_df = (first_condition_values - second_condition_values).dropna()
    n_pairs = int(paired_df.shape[0])
    degrees_of_freedom = n_pairs - 1

    ttest_result = stats.ttest_rel(first_condition_values, second_condition_values, nan_policy="omit")
    p_value_text = text_p_value(ttest_result.pvalue)

    mean_difference = paired_df.mean()
    standard_error = stats.sem(paired_df)

    confidence_interval = stats.t.interval(
        0.95,
        df=degrees_of_freedom,
        loc=mean_difference,
        scale=standard_error,
    )
    ci_half_width = (confidence_interval[1] - confidence_interval[0]) / 2

    # Paired-samples Cohen's d
    sd_difference = paired_df.std(ddof=1)
    paired_sample_cohen_d = mean_difference / sd_difference if sd_difference != 0 else float("nan")

    print(
        f"\n{label}: "
        f"mean={mean_difference:.2f}±{ci_half_width:.2f}, "
        f"t({degrees_of_freedom})={ttest_result.statistic:.2f}, "
        f"{p_value_text};"
    )

    return paired_sample_cohen_d


def print_paired_cohens_d(label: str, paired_sample_cohen_d: float) -> None:
    """Small wrapper to keep prints consistent."""
    print(f"{label}: d={paired_sample_cohen_d:.2f};")


def run_linear_model_and_print(
    dataframe,
    dependent_variable_name: str,
    independent_variable_names: list[str],
    categorical_variable_names: list[str] | None = None,
    label_prefix: str = "",
) -> None:
    """
    Fit OLS model and print:
    - full statsmodels summary
    - intercept β±SE, p
    - each requested coefficient β±SE, p
    """
    categorical_variable_names = categorical_variable_names or []

    rhs_terms = []
    for continuous_name in independent_variable_names:
        rhs_terms.append(continuous_name)
    for categorical_name in categorical_variable_names:
        rhs_terms.append(f"C({categorical_name})")

    formula = f"{dependent_variable_name} ~ " + " + ".join(rhs_terms)
    model = smf.ols(formula, data=dataframe)
    results = model.fit()

    print(f"\n----- lm: {formula}  -----\n")
    print(results.summary())

    ci = results.conf_int(alpha=0.05)  # DataFrame with [0]=lower, [1]=upper

    # Helper function to extract beta, CI half-width, and p-value for a given term
    def beta_ci_halfwidth(term: str):
        beta = results.params.get(term, np.nan)
        p = results.pvalues.get(term, np.nan)
        if term in ci.index:
            lower, upper = float(ci.loc[term, 0]), float(ci.loc[term, 1])
            halfw = (upper - lower) / 2
        else:
            halfw = np.nan
        return beta, halfw, p

    # Intercept
    term = "Intercept"
    intercept_beta, intercept_ci_hw, intercept_p = beta_ci_halfwidth(term)
    print(
        f"\n{label_prefix}LM intercept: β={intercept_beta:.2f}±{intercept_ci_hw:.2f}, {text_p_value(intercept_p)};\n"
    )

    # Categorical coefficients: print all terms for each categorical variable
    for categorical_name in categorical_variable_names:
        for term in results.params.index:
            if term.startswith(f"C({categorical_name})"):
                beta, ci_hw, p = beta_ci_halfwidth(term)
                print(
                    f"{label_prefix}LM effect {term}: "
                    f"β={beta:.2f}±{ci_hw:.2f}, {text_p_value(p)};\n"
                )

    # Continuous IVs
    for continuous_name in independent_variable_names:
        if continuous_name in results.params.index:
            beta, ci_hw, p = beta_ci_halfwidth(continuous_name)
            print(
                f"{label_prefix}LM slope (effect of {continuous_name}): "
                f"β={beta:.2f}±{ci_hw:.2f}, {text_p_value(p)};\n"
            )
