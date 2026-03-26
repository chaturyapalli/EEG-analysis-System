import numpy as np

delta = np.load("data/delta_train.npy")
theta = np.load("data/theta_train.npy")
y = np.load("data/y_train.npy")

delta_mean_normal = np.mean(delta[y == 0])
theta_mean_normal = np.mean(theta[y == 0])

print(delta_mean_normal, theta_mean_normal)