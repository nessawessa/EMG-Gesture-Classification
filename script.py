import numpy as np
import matplotlib.pyplot as py
from scipy.stats import norm

# Helper function for sign changes
def signcross(arr):
    asign = np.sign(arr)
    chksign = np.roll(asign, 1, axis=0) - asign
    signchange = (chksign != 0).astype(int)
    return np.sum(signchange, axis=0)

# Data Preparation
n_trials = 10
n_postures = 8
n_channels = 4

datat = []
for i in range(0, n_postures):
    for j in range(0, n_trials):
        file_name = f'Posture{i}-trial{j+1}.txt'
        A = np.loadtxt(file_name)
        A = A * 1000

        mean_A = np.mean(A, axis=0)
        A = A - mean_A

        A_rms = np.sum(np.power(A, 2), axis=0) / len(A)
        A_std = np.std(A, axis=0)

        A_signcross = signcross(A)
        temp = np.hstack((A_rms, A_std, A_signcross, i))
        datat.append(temp)

data = np.array(datat)
m = data.shape[0]
n_col = data.shape[1]
n_features = n_col - 1

np.random.shuffle(data)

X = data[:, 0:n_col - 1]
Y = data[:, -1]

X_train = X[0:70, :]
Y_train = Y[0:70]
X_test = X[70:, :]
Y_test = Y[70:]

mean_features = np.mean(X_train, axis=0)
std_features = np.std(X_train, axis=0)

X_train = (X_train - mean_features) / std_features
X_test = (X_test - mean_features) / std_features


# Neural Network Implementation
n_inputlayer = len(X_train[0, :])
n_hlayer = 100  # Hidden Nodes
n_outputlayer = n_postures
N = len(X_train)

# One-hot encoding for Y train
Y_one_hot = np.zeros((N, n_outputlayer), dtype='float')
for i in range(N):
    Y_one_hot[i, int(Y_train[i])] = 1.0

# Activation Functions
def ReLU(z):
    return np.maximum(0, z)

def ReLU_prime(z):
    return np.where(z > 0, 1, 0)

# Weight Initialization
W_ij = np.random.normal(0.0, pow(n_inputlayer, -0.5), (n_inputlayer, n_hlayer))
W_jk = np.random.normal(0.0, pow(n_hlayer, -0.5), (n_hlayer, n_outputlayer))

error = []
eta = 0.1  # Learning Rate

# Training Loop
for epoch in range(200000):
    z_nj = np.dot(X_train, W_ij)
    a_nj = ReLU(z_nj)

    z_nk = np.dot(a_nj, W_jk)
    Y_nk = ReLU(z_nk)

    Y_err = Y_nk - Y_one_hot
    temp_nk = np.multiply(Y_err, ReLU_prime(z_nk))
    dw_jk = np.matmul(a_nj.T, temp_nk) / N

    V_err = np.dot(temp_nk, W_jk.T)
    temp_nj = np.multiply(V_err, ReLU_prime(z_nj))
    dw_ij = np.matmul(X_train.T, temp_nj) / N

    W_jk -= eta * dw_jk
    W_ij -= eta * dw_ij

    error.append(np.sum(np.abs(Y_err) / N))

    if epoch % 10000 == 0:
        print(f"Epoch = {epoch}, Error: {error[-1]}")
