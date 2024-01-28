#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns

from numpy.typing import ArrayLike
def line_violin_plotting(save_path:str, dataset_0: ArrayLike, dataset_1: ArrayLike, 
               dataset_2: ArrayLike, ylabel_0: str, ylabel_1: str, ylabel_2: str):
    fig = plt.figure(figsize=(15, 6))
    gs = plt.GridSpec(3, 2, width_ratios=[4, 1])

    ax00 = plt.subplot(gs[0,0])
    ax00.plot(dataset_0, marker='o', markerfacecolor='white')
    ax00.set_ylabel(ylabel_0)
    ax01 = plt.subplot(gs[0,1])
    sns.violinplot(data=dataset_0, ax=ax01, fill=False)

    ax10 = plt.subplot(gs[1,0])
    ax10.plot(dataset_1, marker='o', markerfacecolor='white')
    ax10.set_ylabel(ylabel_1)
    ax11 = plt.subplot(gs[1,1])
    sns.violinplot(data=dataset_1, ax=ax11, fill=False)

    ax20 = plt.subplot(gs[2,0])
    ax20.plot(dataset_2, marker='o', markerfacecolor='white')
    ax20.set_ylabel(ylabel_2)
    ax21 = plt.subplot(gs[2,1])
    sns.violinplot(data=dataset_2, ax=ax21, fill=False)

    plt.tight_layout()
    fig.show()
    fig.savefig(save_path)

