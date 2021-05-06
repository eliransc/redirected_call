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
            total_prob_waiting += u0 * erlang.cdf(w, d + 1, loc=0, scale=mu_11)
        else:
            total_prob_waiting += steady_probs_arr[ind_d] * erlang.cdf(w, d + 1, loc=0, scale=mu_11)

    return total_prob_waiting

def compute_waiting_time_(R,x, mu_11,lam_1 ,lam_ext, ub_v, case_number):

    n = R.shape[0]
    u0 = np.sum(x[:n])
    u1 = x[n:].reshape((1, n))
    steady_probs = []

    for d in tqdm(range(100)):
        if d == 0:
            steady_probs.append(u0)
        else:
            steady_probs.append(np.sum(np.dot(u1, matrix_power(R, d - 1))))
        print(np.sum(np.array(steady_probs)))
        if np.sum(np.array(steady_probs)) > 0.9999:
            break

    steady_probs_arr = np.array(steady_probs)
    w_arr = np.linspace(0, 40, 900)
    waiting_list = []
    for w in w_arr:
        waiting_list.append(waiting_time(w, steady_probs_arr, u0, mu_11))
    print(w_arr)
    print(waiting_list)
    pkl.dump((w_arr, waiting_list), open('../pkl/waiting_list_' + str(ub_v) + '.pkl', 'wb'))


    plt.figure(figsize=(4, 2.5))
    plt.plot(w_arr, waiting_list, label='Out method')
    plt.plot(w_arr, 1 - np.exp(-w_arr * (mu_11 - (lam_1 + lam_ext))), label='Markovian approximation')
    plt.legend()
    plt.xlabel('Waiting time')
    plt.ylabel('Cdf')
    plt.title('Case '+str(case_number))
    plt.show()
