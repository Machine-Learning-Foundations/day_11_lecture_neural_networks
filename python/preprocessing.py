import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(size=(500, 2))/2
# stretch
data = data*np.expand_dims(np.array([1,4]), 0)
# Rotate
#               p
#               |
#               |y
#      x        | 
# o ------------|

# soh
# cah
# toa

# theta = tanh(y/x)
# y = sin(theta)*r
# x = cos(theta)*r
# 
rad = lambda x,y: np.sqrt(x*x+y*y)
data = np.stack([(np.sin(np.arctan2(y,x)-.5)*rad(x,y),
                  np.cos(np.arctan2(y,x)-.5)*rad(x,y)) for x,y in data], 0)
# move to 2,2
data = data + np.expand_dims(np.array([-5,5]), 0)

plt.plot(data[:, 0], data[:, 1], '.', label='input data')
plt.xlim(-10, 10)
plt.ylim(-10, 10)

mean = np.expand_dims(np.mean(data, axis=0), 0)
std = np.expand_dims(np.std(data, axis=0), 0)
data_zero = data - mean
data_norm = data_zero/std

plt.plot(data_norm[:,0], data_norm[:,1], '.', label='normalized data')


cov = data_zero.T@ data_zero/ data_zero.shape[0]
U,s,V = np.linalg.svd(cov)

data_proj = data_zero@U
data_white = data_proj/ np.expand_dims(np.sqrt(s), 0)

plt.plot(data_white[:, 0], data_white[:, 1], '.', label='whitened data')
plt.legend()
plt.show()

print('done')