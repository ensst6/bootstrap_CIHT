# Any changes to the distributions library should be reinstalled with
#  pip install --upgrade .

# For running unit tests, use
# /usr/bin/python -m unittest test

import unittest
import pandas as pd
import numpy as np
from scipy import stats
from bootstrap_CIHT import Bootstrap_CIHT as BS
np.random.seed(42)

# success group is 1000 random nml with mean 0, SD 0.5
# actual mean 0.009666, SD 0.489608
# failure group is 1000 random nml with mean 1, SD 0.5
# actual mean 1.035418, SD 0.498727
smpl1_mean = 0.009666
smpl1_sd = 0.489608
smpl2_mean = 1.035418
smpl2_sd = 0.498727
smpl12_mean = smpl1_mean - smpl2_mean


class TestBSCIHT(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv('nml2.csv')
        self.bs1 = BS(df, 'numbers', num_vars=2, group_col='outcome', group1='failure',\
        group2='success')


    def test_initialization(self):
        self.assertEqual(self.bs1.data_col, 'numbers', 'incorrect data column label')
        self.assertEqual(self.bs1.group1, 'failure', 'incorrect control label')
        self.assertEqual(self.bs1.group2, 'success', 'incorrect expmtl grp label')
        self.assertEqual(self.bs1.df.shape[0], 2000, 'length of df incorrect')

    def test_bootstrap(self):
        exp_mean, ctrl_mean, diffs = self.bs1.get_bootstrap_sample()
        self.assertEqual(len(exp_mean), self.bs1.samples, 'length of exp array incorrect')
        self.assertEqual(len(ctrl_mean), self.bs1.samples, 'length of ctrl array incorrect')
        self.assertEqual(len(diffs), self.bs1.samples, 'length of diff array incorrect')
        # sampling distrib means should be close to the sample means
        self.assertEqual(round(np.mean(exp_mean), 2), round(smpl1_mean, 2), 'exp mean disagrees at 2 places')
        self.assertEqual(round(np.mean(ctrl_mean), 2), round(smpl2_mean, 2), 'ctrl mean disagrees at 2 places')
        self.assertEqual(round(np.mean(diffs), 2), round(smpl12_mean, 2), 'diff mean disagrees at 2 places')

    def test_CIcalc(self):
        exp_mean, ctrl_mean, diffs = self.bs1.get_bootstrap_sample()
        CIs = self.bs1.calculate_CI(diffs)
        Z_lower = -1.960 #2.5th %ile
        Z_upper = 1.960 #97.5th %ile
        # CI = mean + Z*sigma; for sampling distrib, sigma = SD(sample)/sqrt(n)
        # failed with nominal #s; try using actuals
        # for difference in means, mean of sampling distrib is mean2 - mean1
        # sigma = sqrt(sigma1^2/n1+sigma2^2/n2)
        # using the nominal data fails, using the actual data works
        smpl12_sd = np.sqrt((smpl1_sd**2+smpl2_sd**2)/1000)
        diff_CI_low = smpl12_mean + Z_lower*smpl12_sd # = 2*0.5^2/1000
        diff_CI_high = smpl12_mean + Z_upper*smpl12_sd # = 2*0.5^2/1000

        self.assertEqual(round(CIs[0],2), round(diff_CI_low, 2), 'lower ctrl CI disagrees at 2 places')
        self.assertEqual(round(CIs[1],2), round(diff_CI_high, 2), 'upper ctrl CI disagrees at 2 places')

    def test_hypothesis_test(self):
        exp_mean, ctrl_mean, diffs = self.bs1.get_bootstrap_sample()
        p_test = self.bs1.calculate_hypothesis_test(diffs)
        # use a 2-sample t-test
        # for equal n combined var = (var1^2+var2^2)/2, so for equal vars, it's just combinved var = var = 0.25
        # t= (mean2-mean1)/sd*sqrt(1/n1+1/n2)
        t_value = smpl12_mean/(0.5*np.sqrt(2/1000))
        # df = n1+n2-2 = 998
        p_calc = stats.t.sf(np.abs(t_value), 998)*2
        print('test, calc', p_test, p_calc)
        self.assertEqual(round(p_test, 3), round(p_calc, 3), 'p-values disagree at 3 places')

if __name__ == '__main__':
    unittest.main()
