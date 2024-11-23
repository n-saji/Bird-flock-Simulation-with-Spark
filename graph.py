import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# multi = [0.432,1.466,3.438,18.9,77.052,386.267]
loop = [0.18,0.96,2.8,350.29,847.21]
spark = [0.49,0.35,0.31,0.38,0.4]
sentences = [1000, 5000, 10000, 50000, 100000]

plt.plot(sentences,loop, label='Non-Spark',marker='o')
# plt.plot(sentences,multi, label='Multi-Processing',marker='*')
plt.plot(sentences,spark, label='Spark',marker='^')
plt.legend()
plt.xlabel('Number of Scentences')
plt.ylabel('Execution Time (seconds)')
plt.xscale('linear')
plt.yscale('log')
plt.grid(True, which='both', linestyle='-.', linewidth=0.5)
plt.legend(fontsize=12)
plt.xticks(sentences, fontsize=10)
plt.yticks(fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()