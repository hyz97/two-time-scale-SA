import numpy as np
import matplotlib.pyplot as plt 
import os

def pow2(x):
    abs = np.abs(x)
    tmp = abs <= 1
    tmp2 = tmp * abs ** 2 / 2 + (1 - tmp) * (abs - 1 / 2)
    return np.sign(x) * tmp2

def nabla_pow2(x):
    abs = np.abs(x)
    tmp = abs <= 1
    return tmp * x + (1 - tmp) * np.sign(x)

ystar = -0.4501836112948735
xstar = ystar - pow2(ystar)
c_bias = 10

def f_xy(x, y):
    return (x + pow2(y)) ** 2 + np.sin(x + pow2(y))

def nabla_x_f(x, y):
    return 2 * (x + pow2(y)) + np.cos(x + pow2(y))

def nabla_xx_f(x, y):
    return 2 - np.sin(x + pow2(y))

def nabla_xy_f(x, y):
    return 2 * nabla_pow2(y) - np.sin(x + pow2(y)) * nabla_pow2(y)

def g_xy(x, y):
    return (x + pow2(y)) ** 2 + y ** 2 + np.sin(y)

def nabla_x_g(x, y):
    return 2 * (x + pow2(y))

def nabla_y_g(x, y):
    return 2 * (x + pow2(y)) * nabla_pow2(y) + 2 * y + np.cos(y)

def F_xy(x, y):
    return nabla_x_f(x, y)

def G_xy(x, y):
    return nabla_y_g(x, y) - nabla_xy_f(x, y) / nabla_xx_f(x, y) * nabla_x_g(x, y)

# def F_xy_noisy(x, y, var1, var2):
#     dim = np.shape(x)[0]
#     noise_f_xx = np.sqrt(var1) * np.random.randn(dim)
#     noise_f_xy = np.sqrt(var2) * np.random.randn(dim)
#     return nabla_x_f(x, y) + noise_f_xx * x + noise_f_xy * y

# def G_xy_noisy(x, y, var1, var2, var3, var4, n_sample):
#     dim = np.shape(x)[0]
#     noise_g_x = np.sqrt(var1) * np.random.randn(dim)
#     noise_g_y = np.sqrt(var2) * np.random.randn(dim)
#     p = np.random.randint(n_sample + 1, size=dim)
#     noise_f_xx = np.sqrt(var3) * np.random.randn(n_sample, dim)
#     noise_f_xy = np.sqrt(var4) * np.random.randn(dim)
#     nabla_xy_ff = nabla_xy_f(x, y)
#     nabla_xx_ff = nabla_xx_f(x, y)
#     nabla_x_gg = nabla_x_g(x, y)
#     nabla_y_gg = nabla_y_g(x, y)

#     ind = np.tile(np.expand_dims(np.arange(n_sample), axis=1), (1, dim)) < p
#     coef = 1 / (3 + var)
#     tmp = 1 - (nabla_xx_ff + noise_f_xx) * coef
#     tmp = ind * tmp + 1 - ind
#     tmp = np.prod(tmp, axis=0)
#     inv_est = (n_sample + 1) * coef * tmp
#     tmp2 = (nabla_xy_ff + noise_f_xy) * (nabla_x_gg + noise_g_x) * inv_est
#     grad = nabla_y_gg + noise_g_y - tmp2
#     bias = np.abs(nabla_xy_ff * nabla_x_gg * (1 / nabla_xx_ff - inv_est))
#     return [grad, bias]

def H_y(y):
    return ystar - pow2(y)


# para_list = [ (0.5, 1.0), (0.4, 1.0), (0.3, 1.0), (0.2, 1.0), (0.1, 1.0) ]
# alpha_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
# beta_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
alpha_list = [0.7, 0.6]
beta_list = [1.0, 0.9]
# beta_list = [0.9]

# eta1_list = [10, 3, 1, 0.3, 0.1]
# eta2_list = [10, 3, 1, 0.3, 0.1]
eta1_list = [3, 1, 0.3, 0.1]
eta2_list1 = [3, 1, 0.3, 0.1]
eta2_list2 = [3, 1]

var1 = 1 ** 2
var2 = 1 ** 2


n_ite = 100000
dim = 1000

store_path = '../TwoTimeScale_data/exmp51_bias2/' 

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
        to_store_list = []
        if beta < 0.95:
            eta2_list = eta2_list1
        else:
            eta2_list = eta2_list2
        for eta_ini1 in eta1_list:
            for eta_ini2 in eta2_list:
                print('eta', eta_ini1, eta_ini2)
                x = x0
                y = y0
                # print(x0, y0)
                for i in range(n_ite):
                    n_sample = 2 * int(np.log(i+1)) + 1
                    eta_alpha = eta_ini1 / (i+1) ** alpha
                    eta_beta = eta_ini2 / (i+1) ** beta
                    noise_x = np.sqrt(var1) * np.random.randn(dim)
                    noise_y = np.sqrt(var2) * np.random.randn(dim)
                    bias_y = - np.random.randn(dim) * np.sqrt(eta_beta) * np.sign(x) * c_bias
                    x_next = x - eta_alpha * (F_xy(x, y) + noise_x)
                    y_next = y - eta_beta * (G_xy(x, y) + noise_y + bias_y)
                    x = x_next
                    y = y_next
                    x_inner = x - H_y(y)
                    if i % (n_ite // 10) == 0:
                        print(np.linalg.norm(y - ystar) ** 2 / dim)

                x_norm = np.linalg.norm(x - xstar) ** 2 / dim
                y_norm = np.linalg.norm(y - ystar) ** 2 / dim
                x_inner_norm = np.linalg.norm(x_inner) ** 2 / dim

                to_store_list.append([eta_ini1, eta_ini2, x_norm, y_norm, x_inner_norm])
        np.save(path + 'grid.npy', np.array(to_store_list))