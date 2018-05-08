import warnings
from subprocess import check_output

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import Imputer

color = sns.color_palette()
sns.set_style('darkgrid')


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))


print(check_output(['ls', './data']).decode('utf8'))

# dataset = pd.read_csv('data/test.csv')
