import pandas as pd
import pickle as pkl
import os
from utils_ph import *
import numpy as np


def compute_ph_matrix(result, mu_0, mu_1, lam_0,lam_1, path_ph):



    result['ph_size'] = result['mu0'] + result['lam0lam1'] + result['lam0lam1mu0'] + 1
    ts = result['ph_size'].sum()
    probs = np.array(result['prob'])
    ph_size = np.array(result['ph_size'])
    prob_arr = np.zeros((ts, 1))
    initial_ind = []
    for ind_prob, prob in enumerate(probs):
        #     print(ind_prob, prob, np.sum(ph_size[:ind_prob]))
        prob_arr[np.sum(ph_size[:ind_prob])] = prob
        initial_ind.append(np.sum(ph_size[:ind_prob]))

    ph = np.zeros((ts, ts))
    for ind, init_ind in enumerate(initial_ind):
        num_mu0 = result.loc[ind, 'mu0']
        num_lam0lam1 = result.loc[ind, 'lam0lam1']
        num_lam0lam1mu0 = result.loc[ind, 'lam0lam1mu0']

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
    pkl.dump((prob_arr, ph), open(path_ph, 'wb'))
