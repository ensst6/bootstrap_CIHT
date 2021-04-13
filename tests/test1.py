# this is template for testing with continuous data
# did both one- and two-variable cases with data from the DAND IntroStats
# Lessson 11-13
# what's here is a two-sided test for height coffee vs non
# background notebook for this is "Simulating from the Null"
import pandas as pd
import numpy as np
from bootstrap_CIHT import Bootstrap_CIHT as BS
np.random.seed(42)
full_df = pd.read_csv('coffee_dataset.csv')

sample1 = full_df.sample(200)

bs1 = BS(data=sample1, data_col='height', num_vars=2, group_col='drinks_coffee',\
group1=False, group2=True)

# this tests initiliziaton & the fill_data method
print(bs1.df.head())

# test the boostrapping and CI calulation
exp_mean, ctrl_mean, diffs = bs1.get_bootstrap_sample()
CIs = bs1.calculate_CI(diffs)

# test plotting histo with CI
bs1.plot_hist_CI(diffs, CIs, bins=20)
bs1.plot_distribs(exp_mean, ctrl_mean, bins=20)

# test the p-value
print(np.std(diffs))
p_val = bs1.calculate_hypothesis_test(diffs)
#
