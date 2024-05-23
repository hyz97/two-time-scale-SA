import numpy as np
import matplotlib.pyplot as plt 
import os

ystar = -0.4501836112948735
xstar = 0

def f(x):
    return x ** 2 + np.sin(x)

def nabla_f(x):
    return 2 * x + np.cos(x)

def H_y(x):
    return nabla_f(y)

# para_list = [ (0.5, 1.0), (0.4, 1.0), (0.3, 1.0), (0.2, 1.0), (0.1, 1.0) ]
# alpha_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
# beta_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
alpha_list = [0.7, 0.6]
beta_list = [1.0]
# alpha_list = [0.6]
# beta_list = [0.9]

var1 = 1 ** 2
var2 = 0 ** 2

n_ite = 100000
dim = 1000

load_path = '../TwoTimeScale_data/exmp4/' 
store_path = '../TwoTimeScale_data/exmp4_b1.0/' 

# for (alpha, beta) in para_list:
for alpha in alpha_list:
    for beta in beta_list:
        print('alpha', alpha, 'beta', beta)
        x0 = np.ones(dim) * 2
        y0 = np.ones(dim) * 2
        para = 'alpha' + str(alpha) + 'beta' + str(beta)
        path = store_path + para + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        grid_infor = np.load(load_path + para + '/grid.npy')
        b_ind = grid_infor[:,1] > 0.9
        grid_infor = grid_infor[b_ind,:]
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
            x_next = x - eta_alpha * (x - nabla_f(y) + noise_x)
            y_next = y - eta_beta * (x)
            x = x_next
            y = y_next
            x_inner = x - H_y(y)
            x_list.append(np.linalg.norm(x - xstar) ** 2 / dim)
            y_list.append(np.linalg.norm(y - ystar) ** 2 / dim)
            x_inner_list.append(np.linalg.norm(x_inner) ** 2 / dim)
        np.save(path + 'x.npy', np.array(x_list))
        np.save(path + 'y.npy', np.array(y_list))
        np.save(path + 'x_inner.npy', np.array(x_inner_list))