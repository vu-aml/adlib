import seaborn as sns
import matplotlib.pyplot as plt

f1_table = [[0.562, 0.679, 0.648, 0.633, 0.708],
            [0.616, 0.646, 0.61, 0.779, 0.739],
            [0.625, 0.754, 0.644, 0.621, 0.744],
            [0.683, 0.702, 0.706, 0.730, 0.709]]
print(min(min(f1_table)))
ax = sns.heatmap(f1_table, vmin=min(min(f1_table))-0.2, vmax=max(max(f1_table))+0.2, cmap="RdYlGn")
plt.show()