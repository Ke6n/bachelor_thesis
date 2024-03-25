#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns

def line_plotting(save_path:str, data):
    plt.figure(figsize=(15, 6))
    sns.lineplot(data, markers=True)
    plt.yticks(range(int(1), int(5) + 2))
    plt.xticks(rotation= 30)
    plt.savefig(save_path)
    plt.show()