import numpy as np
import matplotlib.pyplot as plt 
import os

delta_G = 1.5

def H_y(y):
    return y

def F_xy(x, y):
    return x - H_y(y)

def G_xy(x, y):
    abs = np.abs(x - H_y(y))
    tmp = abs <= 1
    tmp2 = tmp * abs ** delta_G / delta_G + (1 - tmp) * (abs - 1 + 1 / delta_G) 
    return - tmp2 * np.sign(y) + y

para_list = [ (0.7, 1.0), (0.7, 0.9), (0.6, 1.0), (0.6, 0.9) ]

eta1_list = [10, 3, 1, 0.3, 0.1]
eta2_list = [10, 3, 1, 0.3, 0.1]
# eta1_list = [3, 1, 0.3, 0.1]
# eta2_list = [3, 1, 0.3, 0.1]

var1 = 1 ** 2
var2 = 0.1 ** 2

n_ite = 100000
dim = 1000

store_path = '../TwoTimeScale_data/exmp6/' 

for (alpha, beta) in para_list:
    print('alpha', alpha, 'beta', beta)
    x0 = np.ones(dim) * 2
    y0 = np.ones(dim) * 2
    para = 'alpha' + str(alpha) + 'beta' + str(beta)
    path = store_path + para + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    to_store_list = []
    for eta_ini1 in eta1_list:
        for eta_ini2 in eta2_list:
            print('eta', eta_ini1, eta_ini2)
            x = x0
            y = y0
            for i in range(n_ite):
                eta_alpha = eta_ini1 / (i+1) ** alpha
                eta_beta = eta_ini2 / (i+1) ** beta
                noise_x = np.sqrt(var1) * np.random.randn(dim)
                noise_y = np.sqrt(var2) * np.random.randn(dim)
                x_next = x - eta_alpha * (F_xy(x, y) + noise_x)
                y_next = y - eta_beta * (G_xy(x, y) + noise_y)
                x = x_next
                y = y_next
                x_inner = x - H_y(y)

            x_norm = np.linalg.norm(x) ** 2 / dim
            y_norm = np.linalg.norm(y) ** 2 / dim
            x_inner_norm = np.linalg.norm(x_inner) ** 2 / dim

            to_store_list.append([eta_ini1, eta_ini2, x_norm, y_norm, x_inner_norm])
    np.save(path + 'grid.npy', np.array(to_store_list))