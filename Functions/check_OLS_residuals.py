def check_OLS_residuals(ols_model):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    influence = ols_model.get_influence()
    summary_frame = influence.summary_frame()
    summary_frame.sort_values('cooks_d', ascending=False).head() # look for cooks_d > 4/n where n = number of observations (e.g. 4/50 = 0.08, 4/(50*6)= 0.0133)
    a=1
        
    predicted = ols_model.fittedvalues
    residuals = ols_model.resid

    # Plot residual vs fitted
    # Purpose: Check linearity and homoscedasticity.
    # What you want: Random scatter of points around 0, with no visible pattern.
    plt.figure(figsize=(4, 4))
    sns.residplot(x=predicted, y=residuals, lowess=True)
    plt.title('Residuals vs Fitted')
    plt.show()
    a=1

    # Q-Q plot
    # Purpose: Check normality of residuals.
    # What you want: Points should fall roughly on the 45° line.
    plt.figure(figsize=(4, 4))
    sm.qqplot(residuals, line='45', fit=True)
    plt.title("Q–Q Plot of Residuals")
    plt.show()
    a=1

    # Histogram of residuals
    plt.figure(figsize=(4, 4))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.show()
    a=1

