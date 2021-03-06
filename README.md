# bootstrap-CIHT
A Python module to calculate confidence intervals and perform hypothesis tests on the mean or proportion of one or two groups using bootstrap sampling.  
Selected (hopefully useful) visualizations of the data and null distribution are provided via [matplotlib](https://matplotlib.org).

## Index
[Description](#description)  
[Parameters](#parameters)  
[Attributes](#attributes)  
[Methods](#methods)  
[Usage](#usage)  
[Dependencies](#depend)  
[Installation](#install)  
[Example](#example)  
[Additional Tests](#addtests)  
[History](#history)  
[License](#license)  

### <a id="description"></a>Description
 *class* bootstrap_CIHT.**Bootstrap_CIHT**_(data, data_col, num_vars=1, null_mean=0.0, group_col=None, group1=None, group2=None, samples=10000, alpha=0.05, h_sides=2, h1_dir=None)_

#### <a id="parameters"></a>Parameters
 **data (Pandas dataframe):**  
Unprocessed user data. Should contain at least a column with the data to analyze. For
proportions, data should be coded as 0 or 1. If a two-group comparison, should also have
a column containing group membership labels.

**data_col (string):**  
Name of column containing the data to be analyzed

**num_vars (int):**  
Number of groups (1 or 2); default 1

**null_mean(float):**  
For one-group test, comparison mean or proportion; default 0.0  
Should be left at 0.0 for two-group comparisons.

**group_col (string):**  
 For two-group comparison, name of column with group labels

**group1 (int, bool, string):**  
For two-group comparison, label for first (control) group

**group2 (int, bool, string):**  
For two-group comparison, label for second (experiment) group

**samples (int):**  
Number of bootstrap samples; default 10000

**alpha (float):**  
Sets size of confidence interval (100*(1-alpha)); default 0.05

**h_sides (int):**  
One- or two-sided confidence interval & hypothesis test; default 2

**h1_dir (string):**  
Inequality direction for 1-sided alternative hypothesis:  
'greater' -> experiment parameter > control parameter  
'less' -> experiment parameter < control parameter  
None (default) -> 2-sided comparison

#### <a id="attributes"></a>Attributes:
**df (Pandas dataframe):**   
Dataframe derived from user-supplied `data` for use in analysis  

#### <a id="methods"></a>Methods:
**fill_data**_()_  
Extract data for analysis from user-supplied dataframe. Performed automatically on instantiation.  
_**Args:**_  
None  
_**Returns:**_   
**df (Pandas dataframe):**  
Dataframe containing data to be analyzed ("data" column) and group membership ("group" column) for two-group comparison

**get_bootstrap_sample**_()_  
Create sampling distributions using bootstrapping.  
_**Args:**_  
None  
_**Returns:**_   
_For one-group analysis:_  
**experiment_mean (numpy array):**  
Bootstrapped means of the group of interest  
_For two-group comparison:_  
**experiment_mean (numpy array):**   
Bootstrapped means for experiment group  
**control_mean (numpy array):**   
Bootstrapped means for control group  
**diffs (numpy array):**   
Differences in means of bootstrapped samples

**calculate_CI**_(means)_  
Compute and print 100\*(1-alpha) confidence intervals for the relevant sampling distribution. For a one-group case, this is for the mean of the data. For a two-group case, it is for the difference in means.  
**_Args:_**  
**means (numpy array):**  
Contains either the bootstrapped mean (for a single group) or difference in means (for two groups)  
_**Returns:**_   
**CI (tuple):**  
Upper and lower confidence limits

**calculate_hypothesis_test**_(means)_  
Calculate p-value for a 1- or 2-side hypothesis test comparing the sample mean to the null mean. For two-group comparisons, difference in means is compared to a null mean of zero.  
Plots the observed statistic on a histogram of the null distribution.   
**_Args:_**  
**means (numpy array):**  
Contains either the bootstrapped mean (for a single group) or difference in means (for two groups)  
_**Returns:**_  
**p_value (float):**  
Probability of the observed statistic given the null hypothesis

**plot_hist_CI**_(means, CI, bins=10)_  
Plot the sampling distribution with confidence intervals as vertical red lines. For the single-group case, also plots the null mean as a vertical green line.  
**_Args:_**  
**means (numpy array):**  
Contains either the bootstrapped mean (for a single group) or difference in means (for two groups)  
**CI (tuple):**  
Confidence limits.  
**bins (int):**  
Number of bins in the histogram; default 10  
**_Returns:_**  
None

**plot_distribs**_(experiment_mean, control_mean, bins=10)_  
For a two-group comparison, plot histograms of the control and experimental sampling distributions on a single graph.  
**_Args:_**  
**experiment_mean (np array):**  
Bootstrapped samples of mean for experiment group  
**control_mean (np array):**  
Bootstrapped samples of mean for control group  
**bins (int):**   
Number of bins in the histogram; default 10  
**_Returns:_**  
None

### <a id="usage"></a>Usage
#### <a id="depend"></a>Dependencies
`numpy : 1.19.1`  
`pandas : 1.1.3`  
`matplotlib : 3.3.1`  
additionally for unit tests:  
`scipy 1.5.2`

Developed & tested in `Python 3.8.5`

#### <a id="install"></a>Installation
Assuming working `numpy`, `pandas`, and `matplotlib`:  

`> pip install bootstrap_CIHT`  

#### <a id="example"></a>Example
This uses a 'toy' dataset (available in the `tests` folder) with two groups: "success" and "failure". The success group is 1000 samples from a normal distribution with mean 0.0 and standard deviation 0.5. The failure groups is 1000 samples from a normal distribution with a mean of 1.0 and standard deviation of 0.5.

Note that each pop-up plot window will need to be closed for the script to continue.

```python
>>> import pandas as pd
>>> import numpy as np
>>> from bootstrap_CIHT import Bootstrap_CIHT as BS
>>> np.random.seed(42) #for reproducibility
>>> df = pd.read_csv('nml2.csv')
>>> bs1 = BS(df, 'numbers', num_vars=2, group_col='outcome', group1='failure', group2='success')
>>> bs1.df.head() #the processed data
       data    group
0  0.248357  success
1 -0.069132  success
2  0.323844  success
3  0.761515  success
4 -0.117077  success
>>> bs1.df.tail()
          data    group
1995  1.535075  failure
1996  0.986739  failure
1997  0.559063  failure
1998  0.918467  failure
1999  0.627549  failure
>>> exp_mean, ctrl_mean, diffs = bs1.get_bootstrap_sample()
>>> CIs = bs1.calculate_CI(diffs)
2-sided CI limits: 2.5 97.5
95% 2-sided CI for differences in means of numbers: (-1.0697323775784746, -0.9825440488702097)

>>> p_val = bs1.calculate_hypothesis_test(diffs)
experimental group: success; mean: 0.0097
control group: failure; mean: 1.0354
difference of means, sampling distribution SD: -1.0258, 0.0223
H0: Difference in means success vs failure = 0
H1: Difference in means success vs failure <> 0

p-value: 0.000000

>>> bs1.plot_hist_CI(diffs, CIs, bins=20)
>>> bs1.plot_distribs(exp_mean, ctrl_mean, bins=20)
```

The plot produced by `calculate_hypothesis_test` is:
![Figure 1](./images/Null+mean.png)

The plot from `plot_hist_CI` is:
![Figure 2](./images/diffs.png)

Finally, `plot_distribs` produces:
![Figure 3](./images/distribs.png)

#### <a id="addtests"></a>Additional Tests
The `tests` folder contains unit tests in `test.py`. These use the same toy data as above and are run via:  
`>python -m unittest test.py`   
There are also three other files (`test0.py`, `test1.py`, `test2.py`, `test3.py`) that reproduce exercises from a Udacity Introduction to Statistics course I previously took. The source `.csv` files are included, as are the Jupyter notebooks with the original exercises and results.  
Note that the exercise in `test0.py` appears to produce results that vary a fair amount with each re-sampling, so you may get inconsistent CIs and p-values.

### <a id="history"></a>History
Created April 12, 2021

### <a id="license"></a>License  
[Licensed](license.md) under the [MIT License](https://spdx.org/licenses/MIT.html).
