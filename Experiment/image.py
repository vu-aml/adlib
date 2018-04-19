import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas
import seaborn as sns

matrix = []
with open('FD_result.txt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',',skipinitialspace=True)
    header = next(reader, None)
    for row in reader:
         matrix.append(row)

matrix = [list(i) for i in zip(*matrix)]
print(matrix[4])
print(header)

sns.set()
plt.plot(matrix[0], matrix[7])
plt.show()