import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# multi = [0.432,1.466,3.438,18.9,77.052,386.267]
loop = [0.0037 ,0.0288 ,0.6647 ,3.2473]
spark = [0.0048 ,0.0037 ,0.0106 ,0.0217 ]
birds = [200, 1000, 5000, 10000]

plt.plot(birds,loop, label='Non-Spark',marker='o')
# plt.plot(sentences,multi, label='Multi-Processing',marker='*')
plt.plot(birds,spark, label='Spark',marker='^')
plt.legend()
plt.xlabel('Number of Birds')
plt.ylabel('Execution Time (seconds)')
plt.xscale('linear')
plt.yscale('log')
plt.grid(True, which='both', linestyle='-.', linewidth=0.5)
plt.legend(fontsize=12)
plt.xticks(birds, fontsize=10)
plt.yticks(fontsize=10)

# Show the plot
plt.tight_layout()
plt.show()