# this is template for testing with continuous data
# did both one- and two-variable cases with data from the DAND IntroStats
# Lessson 11-13
# what's here is a two-sided test for height coffee vs non
# background notebook for this is "What is the impact of sample size-solution (1)"
# NB: the distribution here fluctuates widely with re-runs.
import pandas as pd
import numpy as np
from bootstrap_CIHT import Bootstrap_CIHT as BS
np.random.seed(42)
full_df = pd.read_csv('coffee_dataset.csv')

sample1 = full_df.sample(300)

bs1 = BS(data=sample1, data_col='height', num_vars=1, null_mean=67.60, h_sides=1, h1_dir='greater')

# this tests initiliziaton & the fill_data method
print(bs1.df.head())

# test the boostrapping and CI calulation
exp_mean = bs1.get_bootstrap_sample()
CIs = bs1.calculate_CI(exp_mean)

# test plotting histo with CI
bs1.plot_hist_CI(exp_mean, CIs, bins=20)
#bs1.plot_distribs(exp_mean, ctrl_mean, bins=20)

# test the p-value
print(np.std(exp_mean))
p_val = bs1.calculate_hypothesis_test(exp_mean)
#
