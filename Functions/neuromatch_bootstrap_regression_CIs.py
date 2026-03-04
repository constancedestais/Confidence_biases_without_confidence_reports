''''
source: https://github.com/NeuromatchAcademy/course-content/blob/main/tutorials/W1D2_ModelFitting/solutions/W1D2_Tutorial3_Solution_d73b40e4.py

This can bootstrap regression confidence intervals for simple linear regression: y = theta * x + noise

'''

import numpy as np
from sklearn.utils import resample

def _resample_with_replacement(x, y):
  """Resample data points with replacement from the dataset of `x` inputs and
  `y` measurements.

  Args:
    x (ndarray): An array of shape (samples,) that contains the input values.
    y (ndarray): An array of shape (samples,) that contains the corresponding
      measurement values to the inputs.

  Returns:
    ndarray, ndarray: The newly resampled `x` and `y` data points.
  """

  # Get array of indices for resampled points
  sample_idx = np.random.choice(len(x), size=len(x), replace=True)

  # Sample from x and y according to sample_idx
  x_ = x[sample_idx]
  y_ = y[sample_idx]

  return x_, y_


def _solve_normal_eqn(x, y):
  """Solve the normal equations to produce the value of theta_hat that minimizes
    MSE.

    Args:
    x (ndarray): An array of shape (samples,) that contains the input values.
    y (ndarray): An array of shape (samples,) that contains the corresponding
      measurement values to the inputs.
    thata_hat (float): An estimate of the slope parameter.

  Returns:
    float: the value for theta_hat arrived from minimizing MSE
  """
  theta_hat = (x.T @ y) / (x.T @ x)
  return theta_hat

def _bootstrap_estimates(x, y, n=2000):
  """Generate a set of theta_hat estimates using the bootstrap method.

  Args:
    x (ndarray): An array of shape (samples,) that contains the input values.
    y (ndarray): An array of shape (samples,) that contains the corresponding measurement values to the inputs.
    n (int): The number of estimates to compute

  Returns:
    ndarray: An array of estimated parameters with size (n,)
  """
  theta_hats = np.zeros(n)

  # Loop over number of estimates
  for i in range(n):

    # Resample x and y
    x_, y_ = _resample_with_replacement(x, y)
    # Compute theta_hat for this sample
    theta_hats[i] = _solve_normal_eqn(x_, y_)

  return theta_hats


def confidence_interval_from_bootstrap(x, y, n=2000, CI_level=95):
    '''
    I inferred this function from the code in the tutorial which uses the theta_hats to add Confidence Intervals in the plotting code like this: 
        ax.axvline(np.percentile(theta_hats, 50), color='r', label='Median')
        ax.axvline(np.percentile(theta_hats, 2.5), color='b', label='95% CI')
        ax.axvline(np.percentile(theta_hats, 97.5), color='b')
    '''
    
    theta_hats = _bootstrap_estimates(x, y, n=2000)

    lower_bound_level = (100 - CI_level) / 2
    upper_bound_level = 100 - lower_bound_level

    lower_bound = np.percentile(theta_hats, lower_bound_level)
    upper_bound = np.percentile(theta_hats, upper_bound_level)

    return lower_bound, upper_bound

  

def unit_test():
    """Unit test for the following functions: 
    - _bootstrap_estimates
    - _resample_with_replacement, 
    - _solve_normal_eqn
    """
    #--------- simulate data ---------
    np.random.seed(121)
    # set parameters
    theta = 1.2
    n_samples = 15
    # Draw x and then calculate y
    x = 10 * np.random.rand(n_samples)  # sample from a uniform distribution over [0,10)
    noise = np.random.randn(n_samples)  # sample from a standard normal distribution
    y = theta * x + noise

    #--------- simulate data ---------
    # Set random seed
    np.random.seed(123)
    # Get bootstrap estimates
    theta_hats = _bootstrap_estimates(x, y, n=2000)
    #print(theta_hats[0:5])

    # --------- check first five estimates ---------
    expected = [1.27550888, 1.17317819, 1.18198819, 1.25329255, 1.20714664]
    assert np.allclose(theta_hats[0:5], expected, atol=1e-6), "Unit test failed!"


unit_test()