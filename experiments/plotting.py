#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.typing import ArrayLike

def __outliers(arr: ArrayLike):
    q1, q3 = np.percentile(arr, [25, 75])
    whisk_l = q1 - (q3 - q1) * 1.5
    whisk_h = q3 + (q3 - q1) * 1.5
    return arr[(arr < whisk_l)|(arr > whisk_h)]

def line_violin_plotting(save_path:str, ylabels: list[str], *errors: ArrayLike):
    fig = plt.figure(figsize=(15, 6))
    ncols = 2
    gs = plt.GridSpec(len(errors), ncols, width_ratios=[3.5, 1.5])
    gs_x = 0
    gs_y = 0
    for e in errors:
        ax0 = plt.subplot(gs[gs_y,gs_x])
        ax0.plot(e, marker='o', markerfacecolor='white')
        ax0.set_ylabel(ylabels[gs_y])
        plt.ylim(int(min(e)))
        ax1 = plt.subplot(gs[gs_y,gs_x + 1])
        sns.violinplot(data=e, ax=ax1, fill=False, cut=0)
        sns.scatterplot(data=__outliers(e), markers='o', ax=ax1)
        plt.ylim(int(min(e)))
        gs_y += 1
    
    plt.tight_layout()
    fig.show()
    fig.savefig(save_path)