from data_loader import get_data
import numpy as np

x_data, y_data = get_data()

weights = np.random.rand(3, 1)

count = 0
for xi, yi in zip(x_data, y_data):
    print(xi, yi)
    print(xi.shape)
    print(weights.shape)
    count += 1
    if count == 5:
        break
