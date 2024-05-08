#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns

def heatmap_plotting(save_path:str, data):
    plt.figure(figsize=(15, 15))
    ax = sns.heatmap(data=data, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Correlation', fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.xlabel('Metrics', fontsize= 20)
    plt.ylabel('Metrics', fontsize= 20)
    plt.savefig(save_path)
    plt.show()

def scatter_matrix_plotting(save_path:str, data):
    plt.figure(figsize=(15, 15))
    sns.pairplot(data=data, kind='scatter', diag_kind='kde')
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.xlabel('Metrics', fontsize= 20)
    plt.ylabel('Metrics', fontsize= 20)
    plt.savefig(save_path)
    plt.show()