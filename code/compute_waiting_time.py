from tqdm import tqdm
from butools.ph import *
from butools.map import *
from butools.queues import *
from butools.mam import *
from butools.dph import *
from scipy.linalg import expm, sinm, cosm
import pickle as pkl
from numpy.linalg import matrix_power
import numpy as np
from scipy.stats import erlang
import matplotlib.pyplot as plt

def waiting_time(w, steady_probs_arr, u0, mu_11):
    total_prob_waiting = 0
    for ind_d, d in enumerate(range(steady_probs_arr.shape[0])):
        if d == 0:
            total_prob_waiting += u0 * erlang.cdf(w, d + 1, loc=0, scale=1/mu_11)
        else:
            total_prob_waiting += steady_probs_arr[ind_d] * erlang.cdf(w, d + 1, loc=0, scale=1/mu_11)

    return total_prob_waiting

def compute_waiting_time_(R,x, mu_11,lam_1 ,lam_ext, ub_v, case_number):

    n = R.shape[0]
    u0 = np.sum(x[:n])
    u1 = x[n:].reshape((1, n))
    steady_probs = []
    # flag = 0
    for d in tqdm(range(150)):
        if d == 0:
            steady_probs.append(u0)
        else:
            steady_probs.append(np.sum(np.dot(u1, matrix_power(R, d - 1))))
        # print(np.sum(np.array(steady_probs)))


        if np.sum(np.array(steady_probs)) > 0.9999:
            break
    print('The total prob is: ', np.sum(np.array(steady_probs)))
    flag = 0
    steady_probs_arr = np.array(steady_probs)
    w_arr = np.linspace(0, 40, 900)
    waiting_list = []
    for ind_w, w in enumerate(w_arr):
        waiting_list.append(waiting_time(w, steady_probs_arr, u0, mu_11))

        if (waiting_list[-1]>0.9) & (flag == 0):
            index_90 = ind_w
            flag = 1

    # print(index_90)

    # print(w_arr)
    # print(waiting_list), ,lam_ext
    pkl.dump((w_arr, waiting_list), open('../pkl/waiting_list_'+str(lam_1) + '_'+str(mu_11) + '_'+str(lam_ext) + '_' + str(ub_v) + '.pkl', 'wb'))

    print('The 90th percentile of our method is: ', w_arr[index_90])
    print('The 90th percentile of the poisson process is: ', -np.log(1-waiting_list[index_90])/(mu_11 - (lam_1 + lam_ext)))
    print('The exact percentile is: ', waiting_list[index_90])

    plt.figure(figsize=(4, 2.5))
    plt.plot(w_arr, waiting_list,linewidth=3, alpha  = 0.9, linestyle='dashed',color = 'blue',  label='Our method')
    plt.plot(w_arr, 1 - np.exp(-w_arr * (mu_11 - (lam_1 + lam_ext))),  linewidth=3, alpha = 0.5, color = 'green', label='Markovian approximation')
    plt.legend()
    plt.xlabel('Waiting time')
    plt.ylabel('Cdf')
    plt.title('Case '+str(case_number))
    plt.savefig('../figs/Case '+str(case_number)+'.png')
    # plt.show()
