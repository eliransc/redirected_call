import pickle as pkl
import numpy as np
import pandas as pd
import math
from scipy.linalg import expm
from scipy.integrate import quad

def create_t_1_probs(df_path, lam_0, lam_1, mu_0, mu_1, t_prob_path, t1_path_a, t1_path_b, bayes_prob_a, bayes_prob_b , t_shape):

    t_prob = pkl.load(open(t_prob_path, 'rb'))

    df = pkl.load(open(df_path, 'rb'))

    max_v = df['v'].max()

    df = df.loc[df['c'] <= df['v'], :]
    df.reset_index(drop=True)

    t_1_dict = {}
    marg_Bayes_prob = {}

    for num0 in df['num_mu_0'].unique():
        for num1 in range(1, 3):
            print(num0, num1)
            print(df.loc[(df['num_mu_0'] == num0) & (df['num_mu_1'] == num1), 'baysian_prob'].sum())
            marg_Bayes_prob[str(num0)+'_'+str(num1)] = df.loc[(df['num_mu_0'] == num0) & (df['num_mu_1'] == num1), 'baysian_prob'].sum()
            ph = create_ph_arr_(num0, num1, mu_0, mu_1)
            sums = 0
            t1_curr = []
            for t_1 in range(t_shape):
                cur = quad(create_t1_dens_, 0, 100, args=(t_1, ph, lam_0, lam_1,))[0]
                t1_curr.append(cur)
                #             print(cur)
                sums += cur
            t_1_dict[str(num0) + '_' + str(num1)] = t1_curr
            print(sums)

    pkl.dump(t_1_dict, open(t1_path_a, 'wb'))
    pkl.dump(marg_Bayes_prob, open(bayes_prob_a, 'wb'))

    df = pkl.load(open(df_path, 'rb'))
    df = df.loc[df['c'] == df['v']+1, :]
    df.reset_index(drop=True)

    t1_dict_b = {}
    marg_Bayes_prob_b = {}

    v_vals = df['v'].unique()
    for v in v_vals:
        mu_0_vals = df.loc[df['v'] == v, 'num_mu_0'].unique()
        for num0 in mu_0_vals:
            ph = create_ph_arr_(num0, 1, mu_0, mu_1)
            marg_Bayes_prob_b[str(v) + '_' + str(num0)] = df.loc[
                (df['num_mu_0'] == num0) & (df['v'] == v), 'baysian_prob'].sum()
            t_1_probs = []
            for t_1 in range(t_shape-max_v):
                curr_v = v
                curr_sum = 0
                for t in range(curr_v + 1, curr_v + 1 + t_1 + 1):
                    if t < t_prob.shape[0]:
                        curr_sum += (t_prob[t] / np.sum(t_prob[curr_v + 1:])) * \
                                    quad(create_t1_dens_, 0, 100, args=(t_1 - t + curr_v + 1, ph, lam_0, lam_1,))[0]
                t_1_probs.append(curr_sum)
            t1_dict_b[str(v) + '_' + str(num0)] = t_1_probs

    pkl.dump(t1_dict_b, open(t1_path_b, 'wb'))
    pkl.dump(marg_Bayes_prob_b, open(bayes_prob_b, 'wb'))


def create_t1_dens_(r, t_1, ph, lam_0, lam_1):
    ph0 = -np.dot(ph, np.ones((ph.shape[0], 1)))
    alph = np.zeros(ph.shape[0])
    alph[0] = 1

    numin_1 = np.exp(-(lam_0 + lam_1) * r) * (((lam_0 + lam_1) * r) ** t_1)
    numin_2 = np.dot(np.dot(alph, expm(ph * r)), ph0)

    denom = math.factorial(t_1)

    return (numin_1 * numin_2) / denom


def create_ph_arr_(num_mu0, num_mu1, mu_0, mu_1):
    size = num_mu0 + num_mu1

    curr_arr = np.zeros((size, size))

    for ind_arr in range(num_mu0):
        curr_arr[ind_arr, ind_arr] = -mu_0
        curr_arr[ind_arr, ind_arr + 1] = mu_0

    for ind_arr in range(num_mu0, size):
        curr_arr[ind_arr, ind_arr] = -mu_1
        if ind_arr < size - 1:
            curr_arr[ind_arr, ind_arr + 1] = mu_1

    return curr_arr