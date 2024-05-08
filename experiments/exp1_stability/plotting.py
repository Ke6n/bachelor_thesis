#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def __outliers(arr: np.ndarray):
    q1, q3 = np.percentile(arr, [25, 75])
    whisk_l = q1 - (q3 - q1) * 1.5
    whisk_h = q3 + (q3 - q1) * 1.5
    return arr[(arr < whisk_l)|(arr > whisk_h)]

def violin_plotting(save_path:str, xlabel: str, acc_set: np.ndarray):
    fontsize = 15
    plt.figure(figsize=(15, 6))
    sns.violinplot(x=acc_set, fill=False, cut=0)
    outliers = __outliers(acc_set)
    sns.scatterplot(x=outliers, y=np.zeros(outliers.size), markers='o')
    plt.xlim(xmin=int(min(acc_set)))
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel('Count', fontsize = fontsize)
    plt.tick_params(axis='both', labelsize=fontsize-2)

    plt.savefig(save_path)
    plt.show()