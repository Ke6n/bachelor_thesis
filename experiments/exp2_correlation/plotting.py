#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns

def heatmap_plotting(save_path:str, data):
    plt.figure(figsize=(15, 15))
    sns.heatmap(data=data, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.savefig(save_path)
    plt.show()

def scatter_matrix_plotting(save_path:str, data):
    plt.figure(figsize=(15, 15))
    sns.pairplot(data=data, kind='scatter', diag_kind='kde')
    plt.savefig(save_path)
    plt.show()