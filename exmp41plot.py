import numpy as np
import matplotlib.pyplot as plt 
import os

plt.clf()
plt.figure(figsize = (16, 3))
# plt.figure(figsize = (24, 18))
plt.subplots_adjust(hspace=0.3)
# para_list = [ (0.6, 1.0), (0.7, 1.0), (0.8, 1.0), (0.9, 1.0) ]
# para_list = [ (0.5, 1.0), (0.4, 1.0) ]
# alpha_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
# beta_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
alpha_list = [0.7, 0.6]
beta_list = [1.0, 0.9]

load_path = '../TwoTimeScale_data/exmp41/' 
load_path2 = '../TwoTimeScale_data/exmp41_b1.0/' 
j = 0
n_ite = 100000
s_plot = 10
s_slope = 10000

# for (alpha, beta) in para_list:
for alpha in alpha_list:
    for beta in beta_list:
        j = j + 1

        ax = plt.subplot(1, 4, j)
        para = 'alpha' + str(alpha) + 'beta' + str(beta)
        if beta < 0.95:
            path = load_path + para + '/'
        else:
            path = load_path2 + para + '/'
        # x_data = np.load(path + 'x.npy')
        y_data = np.load(path + 'y.npy')
        x_inner_data = np.load(path + 'x_inner.npy')
        subtitle = r'$a=$' + str(alpha) + ', ' + r'$b=$' + str(beta)
        ind = np.arange(n_ite)
        plt.loglog(ind[s_plot:], x_inner_data[s_plot:], label='x', linewidth=0.3)
        plt.loglog(ind[s_plot:], y_data[s_plot:], label='y', linewidth=0.3)
        plt.ylim([1e-6, 1e0])
        
        slope_x, _ = np.polyfit(np.log(ind[s_slope:]), np.log(x_inner_data[s_slope:]), 1)
        slope_y, _ = np.polyfit(np.log(ind[s_slope:]), np.log(y_data[s_slope:]), 1)
        plt.text(1.2e1, 1.5e-5, 'slope of x: ' + str(round(slope_x, 2)))
        plt.text(1.2e1, 5e-6, 'slope of y: ' + str(round(slope_y, 2)))

        leg = plt.legend()
        for line in leg.get_lines():
            line.set_linewidth(1)
        plt.grid()
        plt.xlabel('Iterations', labelpad=0.5)
        if j == 1:
            plt.ylabel(r'$|\hat{x}_t|^2$' + ' or ' + r'$|\hat{y}_t|^2$')
        plt.title(subtitle)

    plt.savefig('./figs/exmp41.pdf', bbox_inches='tight')