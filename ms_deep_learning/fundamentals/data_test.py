import matplotlib.pyplot as plt
import numpy as np


data = [[i] for i in range(0, 20)]
data2 = [[i*2 - 2] if i % 2 == 0 else [i*2 + 5] for i in range(0, 20)]
data = np.array(data)
data2 = np.array(data2)
print(data)
print(data2)

plt.scatter(data, data2)
plt.show()
