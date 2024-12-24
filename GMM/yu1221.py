import numpy as np
import ray
import matplotlib.pyplot as plt
import time
from procedures import *
from utils import *

from matplotlib import rcParams
plt.rcParams['font.weight'] = 'normal'
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'stix'

num_cpus=10
ray.shutdown()
ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

def MMCV(k=5, m=3):
    means = np.zeros((k, m))
    stds = np.zeros((k, m))
    for i in range(1,k+1):
        for j in range(1, m+1):
            means[i-1, j-1] = 0.5*i - j*0.2 - 1
            stds[i-1, j-1] = 16
    return means, stds


def MMIV(k=5, m=3):
    means = np.zeros((k, m))
    stds = np.zeros((k, m))
    for i in range(1,k+1):
        for j in range(1, m+1):
            means[i-1, j-1] = 0.5*i - j*0.2 - 1
            stds[i-1, j-1] = (12+np.sqrt(0.2*i+j))
    return means, stds


def MMDV(k=5, m=3):
    means = np.zeros((k, m))
    stds = np.zeros((k, m))
    for i in range(1,k+1):
        for j in range(1, m+1):
            means[i-1, j-1] = 0.5*i - j*0.2 - 1
            stds[i-1, j-1] = (12+1/(0.2*i+j))
    return means, stds


means, stds = MMCV()
generator = MatrixAlternativeGenerator(means, stds)


seed = 2024090406 # 增加m并不会有太大的变化｜我们只考虑这一种情况，不同的m也需要考虑
n_replications = 4000
evaluator = evaluate_PCS
rng = np.random.default_rng(seed)
Delta = 20
k = 20
n0 = 20
ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

PCS_list = []

for m in [5, 20, 50]:
    for func in [MMCV, MMIV, MMDV]:
        means, stds = func(k, m)
        
        generators = [MatrixAlternativeGenerator(means, stds)  for i in range(n_replications)]
        for n1 in ns:
            print("---------------RUCB Top---------------")
            PCS = parallel_experiments(rng,  generators, evaluator, 
                                       remote_policy=remote_RUCB, args={"n0":n0, "n1":n1, "Delta":Delta})

            PCS_list.append(PCS)

        print(PCS_list)

        generators = [MatrixAlternativeGenerator(means, stds)  for i in range(n_replications)]
        for n1 in ns:
            print("---------------Additive Proportional Top---------------")
            PCS = parallel_experiments(rng,  generators, evaluator, 
                                       remote_policy=remote_AdditivePropROCBA, args={"n0":n0, "n1":n1, "Delta":Delta})

            PCS_list.append(PCS)


        print(PCS_list)


        generators = [MatrixAlternativeGenerator(means, stds)  for i in range(n_replications)]
        for n1 in ns:
            print("---------------ROCBA Top---------------")
            PCS = parallel_experiments(rng,  generators, evaluator, 
                                       remote_policy=remote_ROCBA, args={"n0":n0, "n1":n1, "Delta":Delta})

            PCS_list.append(PCS)

        print(PCS_list)
#     generators = [MatrixAlternativeGenerator(means, stds)  for i in range(n_replications)]
#     for n1 in ns:
#         print("---------------REA  Top---------------")
#         PCS = parallel_experiments(rng,  generators, evaluator, 
#                                    remote_policy=remote_REA, args={"n":n0+n1})

#         PCS_list.append(PCS)

#     print(PCS_list)

results_collection = np.array(PCS_list).reshape(9, 3, len(ns))


labels = [ 'RUCB',"AR-OCBA", "R-OCBA"]

fontsize=18

colors = ["g", "b","r", "c", "k", "c",]
markers = ["D", "o", "^", "p", "D", "s"]
line_styles = ["--","--", "-.", ":","--", "-."]
line_styles = ["--","--", "-.", ":","--", "-."]
fig = plt.figure(figsize=(15,15))

titles = ["Conf. MMCV with $k={}, m={}$".format(20,5), "Conf. MMIV with $k={}, m={}$".format(20,5), 
          "Conf. MMDV with $k={}, m={}$".format(20,5), 
          "Conf. MMCV with $k={}, m={}$".format(100, 5), 
          "Conf. MMIV with $k={}, m={}$".format(100, 5), "Conf. MMDV with $k={}, m={}$".format(100, 5),
         "Conf. MMCV with $k={}, m={}$".format(200, 5), 
          "Conf. MMIV with $k={}, m={}$".format(200, 5), "Conf. MMDV with $k={}, m={}$".format(200, 5)]

fig_plots = [(3,3,1), (3, 3, 2), (3, 3, 3), 
             (3, 3,  4), (3, 3,  5), (3, 3, 6),
            (3, 3,  7), (3, 3,  8), (3, 3, 9)]

for i, fig_plot in enumerate(fig_plots):
    
    ax = fig.add_subplot(*fig_plots[i])

    if i <= 2:
        _ns = (n0 + np.array(ns))
    else:
        _ns = (n0 + np.array(ns))

    results = np.array(results_collection[i])
    for j, result in enumerate(results):
        plt.plot(_ns, result, line_styles[j], markersize=12, color=colors[j], 
                 markerfacecolor="white", markeredgewidth=1.7, marker=markers[j], label=labels[j])

    plt.xlim(_ns[0], _ns[-1])
    xticks = _ns
    xticklabels = _ns
    plt.xticks(xticks, xticklabels, fontsize=fontsize)

    
    plt.ylim(0.3, 0.9)
    yticks = np.arange(0.3, 1.01, 0.1)
    yticklabels = ["%.0f"%(tick*100)+"%" for tick in yticks]
    if i == 0 or i == 3 or i==6:
        plt.yticks(yticks, yticklabels, fontsize=fontsize)
        plt.ylabel("Estimated PCS", size=fontsize+1)
    else:
        plt.yticks(yticks, [""]*len(yticklabels), fontsize=fontsize)
        
    if i ==0 or i==3 or i==6:
        plt.legend(fontsize=30, prop={'family': 'serif',"size":fontsize})
    
    if i == 6 or i == 7 or i == 8:
        plt.xlabel("$c$  in $N=(c+n_0) k m$", size=fontsize+2)

    plt.title(titles[i], fontsize=fontsize, fontweight="bold")
    plt.grid(0.2)

plt.show()

fig.savefig("Comparing_different_OCBA.pdf", bbox_inches='tight')