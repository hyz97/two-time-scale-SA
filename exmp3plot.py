import numpy as np
import matplotlib.pyplot as plt 
import os

plt.clf()
plt.figure(figsize = (16, 3))
plt.subplots_adjust(hspace=0.3)
para_list = [ (0.6, 1.0), (0.7, 1.0), (0.8, 1.0), (0.9, 1.0) ]
# para_list = [ (0.5, 1.0), (0.4, 1.0) ]

load_path = '../TwoTimeScale_data/exmp3/' 
j = 0
n_ite = 100000
s_plot = 10
s_slope = 10000

for (alpha, beta) in para_list:
    j = j + 1

    ax = plt.subplot(1, 4, j)
    para = 'alpha' + str(alpha) + 'beta' + str(beta)
    path = load_path + para + '/'
    # x_data = np.load(path + 'x.npy')
    y_data = np.load(path + 'y.npy')
    x_inner_data = np.load(path + 'x_inner.npy')
    # subtitle = r'$a=$' + str(alpha) + ', ' + r'$b=$' + str(beta)
    subtitle = r'$a=$' + str(alpha)
    ind = np.arange(n_ite)
    plt.loglog(ind[s_plot:], x_inner_data[s_plot:], label='x', linewidth=0.3)
    plt.loglog(ind[s_plot:], y_data[s_plot:], label='y', linewidth=0.3)
    # plt.ylim([5e-6, 5e-1])
    
    slope_x, _ = np.polyfit(np.log(ind[s_slope:]), np.log(x_inner_data[s_slope:]), 1)
    slope_y, _ = np.polyfit(np.log(ind[s_slope:]), np.log(y_data[s_slope:]), 1)
    plt.text(1.2e1, 1.5e-5, 'slope of x: ' + str(round(slope_x, 2)))
    plt.text(1.2e1, 5e-6, 'slope of y: ' + str(round(slope_y, 2)))
    plt.ylim([8e-7, 1e-1])

    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(1)
    plt.grid()
    plt.xlabel('Iterations', labelpad=0.5)
    if j == 1:
        plt.ylabel(r'$|\hat{x}_t|^2$' + ' or ' + r'$|\hat{y}_t|^2$')
    plt.title(subtitle)

plt.savefig('./figs/exmp3.pdf', bbox_inches='tight')