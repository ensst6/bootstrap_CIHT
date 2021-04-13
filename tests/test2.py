# this is template for testing with proportions
# this is from DAND IntroStats Lesson 13 notebook 'completion_rate'
import pandas as pd
import numpy as np
from bootstrap_CIHT import Bootstrap_CIHT as BS
np.random.seed(42)
full_df = pd.read_csv('classroom_actions.csv')


bs1 = BS(data=full_df, data_col='completed', num_vars=2, group_col='group',\
group1='control', group2='experiment', h_sides=1, h1_dir='greater')

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
