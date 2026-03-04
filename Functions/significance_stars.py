import numpy as np

def significance_stars(p: float) -> str:
    '''
    Parameters    ----------
    p : float
        p-value to convert to significance stars
    Returns    -------
    str        significance stars corresponding to p-value to be displayed in figure  

    '''
    if np.isnan(p):
        return "NAN"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    #if p > 0.05 and p < 0.6:
    #    return "~"
    return "ns"