# function which takes as input a p-value number and outputs text
"""Helper to format p-values as text."""

def text_p_value(number_p):
     
    # display 3 significant digits, but if p is smaller than 0.0001, display "p<0.0001" instead of "p=0.0000"
    if number_p < 0.001:
        return "P<0.001"
    else:
        number_p = round(number_p, 3)
        return f"P={number_p:.3f}"
    '''
    # display 2 significant digits, but if p is smaller than 0.01, display with scientific notation and two significant digits (e.g., "p=1.2x10^-5")
    if number_p < 0.01:
        #  display with scientific notation and two significant digits (e.g., "p=1.2x10^-5")
        return f"p={number_p:.2e}"
    else:
        number_p = round(number_p, 2)
        return f"p={number_p:.2f}" 
    '''