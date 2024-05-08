#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns

def line_plotting(save_path:str, data, fontsize = 12):
    plt.figure(figsize=(15, 6))
    sns.lineplot(data, markers=True)
    plt.yticks(range(int(1), int(4) + 2), fontsize = 13)
    plt.xticks(rotation= 20, fontsize = 13, horizontalalignment='right')
    plt.ylabel('Position', fontsize = 16)
    plt.xlabel('Criterions', fontsize = 16)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()