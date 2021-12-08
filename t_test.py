import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from pylab import rcParams
from scipy.stats import f_oneway
from scipy.stats import ttest_ind



#matplotlib inline
warnings.filterwarnings("ignore")
rcParams['figure.figsize'] = 20,10
rcParams['font.size'] = 30
sns.set()
np.random.seed(8)


def plot_distribution(inp):
    plt.figure()
    ax = sns.distplot(inp)
    plt.axvline(np.mean(inp), color="k", linestyle="dashed", linewidth=5)
    _, max_ = plt.ylim()
    plt.text(
        inp.mean() + inp.mean() / 10,
        max_ - max_ / 10,
        "Mean: {:.2f}".format(inp.mean()),
    )
    return plt.figure

ger_sales = np.load("mcts_testClassic.npy")
print("ger_sales",ger_sales)
print("length",len(ger_sales))
# plot_distribution(ger_sales)
fr_sales = np.load("AlphaBetaAgent_testClassic.npy")
# plot_distribution(fr_sales)



# plt.figure()
# ax1 = sns.distplot(fr_sales)
# ax2 = sns.distplot(ger_sales)
# plt.axvline(np.mean(fr_sales), color='b', linestyle='dashed', linewidth=5)
# plt.axvline(np.mean(ger_sales), color='orange', linestyle='dashed', linewidth=5)
# plt.show()

def compare_2_groups(arr_1, arr_2, alpha, sample_size):
    stat, p = ttest_ind(arr_1, arr_2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')

sample_size = 3
ger_sampled = np.random.choice(ger_sales, sample_size)
fr_sampled = np.random.choice(fr_sales, sample_size)
compare_2_groups(ger_sampled, fr_sampled, 0.05, sample_size)

