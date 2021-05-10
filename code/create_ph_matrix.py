import pandas as pd
import pickle as pkl
import os
from utils_ph import *
import numpy as np
from numpy.linalg import matrix_power



def compute_ph_matrix(result, mu_0, mu_1, lam_0,lam_1, path_ph, ub_v):




    result['ph_size'] = result['mu0'] + result['lam0lam1'] + result['lam0lam1mu0'] + 1

    eps = 10 ** (-3.5)
    mu0_avg = round(result.loc[result['prob'] < eps, 'mu0'].mean()) + 1
    lam0lam1_avg = round(result.loc[result['prob'] < eps, 'lam0lam1'].mean()) + 1
    lam0lam1mu0_avg = round(result.loc[result['prob'] < eps, 'lam0lam1mu0'].mean()) + 1

    reduced_result = result.loc[result['prob'] > eps, :].reset_index()
    curr_row = reduced_result.shape[0]
    reduced_result.loc[curr_row, 'event'] = str(mu0_avg) + '_' + str(lam0lam1_avg) + '_' + str(lam0lam1mu0_avg)
    reduced_result.loc[curr_row, 'prob'] = result.loc[result['prob'] < eps, 'prob'].sum()
    reduced_result.loc[curr_row, 'mu0'] = mu0_avg
    reduced_result.loc[curr_row, 'lam0lam1'] = lam0lam1_avg
    reduced_result.loc[curr_row, 'lam0lam1mu0'] = lam0lam1mu0_avg
    reduced_result.loc[curr_row, 'ph_size'] = int(mu0_avg + lam0lam1_avg + lam0lam1mu0_avg+1)

    print('Total lost prob: ', reduced_result.loc[curr_row, 'prob'])

    result = reduced_result.reset_index()

    epsilon = 1-result['prob'].sum()

    mu0_avg_max_v, lam0lam1_avg_max_v, lam0lam1mu0_avg_max_v = pkl.load(open('mean_num_rates_ub_v.pkl', 'rb'))

    curr_row = reduced_result.shape[0]
    result.loc[curr_row, 'event'] = str(mu0_avg) + '_' + str(lam0lam1_avg) + '_' + str(lam0lam1mu0_avg)
    result.loc[curr_row, 'prob'] = epsilon
    result.loc[curr_row, 'mu0'] = mu0_avg_max_v
    result.loc[curr_row, 'lam0lam1'] = lam0lam1_avg_max_v
    result.loc[curr_row, 'lam0lam1mu0'] = lam0lam1mu0_avg_max_v
    result.loc[curr_row, 'ph_size'] = int(mu0_avg_max_v + lam0lam1_avg_max_v + lam0lam1mu0_avg_max_v + 1)



    ts = int(result['ph_size'].sum())
    probs = np.array(result['prob'])
    ph_size = np.array(result['ph_size']).astype(int)
    prob_arr = np.zeros((ts, 1))
    initial_ind = []
    for ind_prob, prob in enumerate(probs):
        #     print(ind_prob, prob, np.sum(ph_size[:ind_prob]))
        prob_arr[np.sum(ph_size[:ind_prob])] = prob
        initial_ind.append(np.sum(ph_size[:ind_prob]))

    ph = np.zeros((ts, ts))
    for ind, init_ind in enumerate(initial_ind):
        num_mu0 = int(result.loc[ind, 'mu0'])
        num_lam0lam1 = int(result.loc[ind, 'lam0lam1'])
        num_lam0lam1mu0 = int(result.loc[ind, 'lam0lam1mu0'])

        for mu_0_ind in range(num_mu0):
            ph[init_ind + mu_0_ind, init_ind + mu_0_ind] = -mu_0
            ph[init_ind + mu_0_ind, init_ind + mu_0_ind + 1] = mu_0

        for lam0lam1_ind in range(num_lam0lam1):
            ph[init_ind + num_mu0 + lam0lam1_ind, init_ind + num_mu0 + lam0lam1_ind] = -(lam_0 + lam_1)
            ph[init_ind + num_mu0 + lam0lam1_ind, init_ind + num_mu0 + lam0lam1_ind + 1] = lam_0 + lam_1

        for lam0lam1mu0_ind in range(num_lam0lam1mu0):
            ph[init_ind + num_mu0 + num_lam0lam1 + lam0lam1mu0_ind, init_ind + num_mu0 + num_lam0lam1 + lam0lam1mu0_ind] \
                = -(lam_0 + lam_1 + mu_0)
            ph[init_ind + num_mu0 + num_lam0lam1 + lam0lam1mu0_ind, init_ind + num_mu0 + num_lam0lam1 + lam0lam1mu0_ind + 1] \
                = lam_0 + lam_1 + mu_0

        ph[init_ind + num_mu0 + num_lam0lam1 + num_lam0lam1mu0, init_ind + num_mu0 + num_lam0lam1 + num_lam0lam1mu0] = -mu_1

    prob_arr = prob_arr.reshape((1, prob_arr.shape[0]))

    PH_minus_2 = matrix_power(ph, -2)
    second_moment = 2 * np.sum(np.dot(prob_arr, PH_minus_2))
    variance = second_moment - (1 / lam_1) ** 2

    print('The true variance is: ', variance)
    print('The markovian variance is:', (1/lam_1)**2)

    pkl.dump((prob_arr, ph), open(path_ph, 'wb'))