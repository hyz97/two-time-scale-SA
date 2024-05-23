import numpy as np
import matplotlib.pyplot as plt 
import os

def H_y(y):
    return y

def F_xy(x, y):
    return x - H_y(y)

def G_xy(x, y):
    tmp = x - H_y(y)
    return - np.abs(tmp) * np.sign(y) + y

para_list = [ (0.7, 1.0), (0.6, 1.0), (0.6, 0.9), (0.7, 0.9)  ]
# para_list = [ (0.7, 0.9) ]

# eta1_list = [10, 3, 1, 0.3, 0.1]
# eta2_list = [10, 3, 1, 0.3, 0.1]

var1 = 1 ** 2
var2 = 0.1 ** 2

n_ite = 1000000
dim = 1000

store_path = '../TwoTimeScale_data/exmp1/' 

for (alpha, beta) in para_list:
    print('alpha', alpha, 'beta', beta)
    x0 = np.ones(dim) * 2
    y0 = np.ones(dim) * 2
    para = 'alpha' + str(alpha) + 'beta' + str(beta)
    path = store_path + para + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    grid_infor = np.load(path + 'grid.npy')
    eta_ind = np.argmin(grid_infor[:,3])
    eta_ini1 = grid_infor[eta_ind,0]
    eta_ini2 = grid_infor[eta_ind,1]
    print(eta_ini1, eta_ini2)
    x_list = []
    y_list = []
    x_inner_list = []
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
        x_list.append(np.linalg.norm(x) ** 2 / dim)
        y_list.append(np.linalg.norm(y) ** 2 / dim)
        x_inner_list.append(np.linalg.norm(x_inner) ** 2 / dim)
    np.save(path + 'x.npy', np.array(x_list))
    np.save(path + 'y.npy', np.array(y_list))
    np.save(path + 'x_inner.npy', np.array(x_inner_list))