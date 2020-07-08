import numpy as np
import matplotlib.pyplot as plt

# make a data with four attributes.
data = [[4, 5, 3, 4], [2, 3, 4, 1], [2, 5, 3, 5], [4, 2, 3, 1], [2, 13, 4, 6]]

data2 = [[54], [45], [41], [42], [46]]
data = np.array(data)
data2 = np.array(data2)
plt.scatter(data[:, 0], data2)
plt.show()
