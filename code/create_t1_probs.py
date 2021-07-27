import pickle as pkl
import numpy as np
import pandas as pd
import math
from scipy.linalg import expm
from scipy.integrate import quad
from utils_ph import *
from tqdm import tqdm

def create_t_1_probs(df_path, lam_0, lam_1, mu_0, mu_1, t_prob_path, h0, t1_path):

    t_prob = pkl.load(open(t_prob_path, 'rb'))

    df = pkl.load(open(df_path, 'rb'))

    max_v = df['v'].max()




    t_1_dict = {}

    curr_inds = df.loc[df['c'] == 0, :].index

    for dist_ind in tqdm(curr_inds):

        num_mu0 = df.loc[dist_ind, 'mu0']
        curr_v = num_mu0
        num_lam0_1 = df.loc[dist_ind, 'lam0lam1']
        num_mu0_lam0_1 = df.loc[dist_ind, 'lam0lam1mu0']
        curr_ph = create_curr_ph_inter(num_mu0, num_lam0_1, num_mu0_lam0_1, lam_0, lam_1, mu_0, mu_1)

        sums = 0.
        t_1 = 0
        t1_curr = []
        while sums < 0.99999:
            curr_sum = 0.
            for t in range(curr_v + 1, curr_v + 1 + t_1 + 1):
                if t < t_prob.shape[0]:
                    curr_k = t_1 - t + curr_v + 1
                    if (t_prob[t] / np.sum(t_prob[curr_v + 1:])) > 0.000001:
                        numirator = (t_prob[t] / np.sum(t_prob[curr_v + 1:])) * (
                                    np.exp(-(lam_0 + lam_1) * h0) * ((lam_0 + lam_1)*h0) ** curr_k)
                        if (numirator > 0.000000001) & (curr_k < 150):
                            curr_sum += numirator/math.factorial(curr_k)
            t_1 += 1
            sums += curr_sum
            t1_curr.append(curr_sum)
        t_1_dict[str(dist_ind)] = t1_curr


    curr_inds = df.loc[df['c'] > 0, :].index
    for dist_ind in tqdm(curr_inds):


        num_0_from_ar = df.loc[dist_ind, 'num_mu_0']
        num_mu0 = df.loc[dist_ind, 'mu0']
        num_lam0_1 = df.loc[dist_ind, 'lam0lam1']
        num_mu0_lam0_1 = df.loc[dist_ind, 'lam0lam1mu0']
        curr_ph = create_curr_ph_inter(num_mu0, num_lam0_1, num_mu0_lam0_1, lam_0, lam_1, mu_0, mu_1)

        ph_before = create_ph_before(num_mu0-num_0_from_ar,num_lam0_1, num_mu0_lam0_1, lam_0, lam_1, mu_0)
        ph_after = create_ph_after(num_mu0, mu_0, mu_1)

        sums = 0
        t1_curr = []

        # for t_1 in range(t_prob.shape[0]):
        t_1 = 0
        while sums < 0.99999:
            cur = quad(create_t1_dens_, 0, h0, args=(h0, t_1, curr_ph, ph_before, ph_after, lam_0, lam_1,))[0]
            t1_curr.append(cur)
            sums += cur
            t_1 += 1
        t_1_dict[str(dist_ind)] = t1_curr


    pkl.dump(t_1_dict, open(t1_path, 'wb'))




    # t_1_dict = {}
    # marg_Bayes_prob = {}
    #
    # for num0 in df['num_mu_0'].unique():
    #     num1 = 1
    #     print(num0, num1)
    #     print(df.loc[(df['num_mu_0'] == num0) & (df['num_mu_1'] == num1), 'baysian_prob'].sum())
    #     marg_Bayes_prob[str(num0)+'_'+str(num1)] = df.loc[(df['num_mu_0'] == num0) & (df['num_mu_1'] == num1), 'baysian_prob'].sum()
    #     ph = create_ph_arr_(num0, num1, mu_0, mu_1)
    #     sums = 0
    #     t1_curr = []
    #     for t_1 in range(t_shape):
    #         cur = quad(create_t1_dens_, 0, 100, args=(t_1, ph, lam_0, lam_1,))[0]
    #         t1_curr.append(cur)
    #         #             print(cur)
    #         sums += cur
    #     t_1_dict[str(num0) + '_' + str(num1)] = t1_curr
    #     print(sums)
    #
    # pkl.dump(t_1_dict, open(t1_path_a, 'wb'))
    # pkl.dump(marg_Bayes_prob, open(bayes_prob_a, 'wb'))
    #
    # df = pkl.load(open(df_path, 'rb'))
    # df = df.loc[df['c'] == 0, :]
    # df.reset_index(drop=True)
    #
    # t1_dict_b = {}
    # marg_Bayes_prob_b = {}
    #
    # v_vals = df['v'].unique()
    # for v in v_vals:
    #     mu_0_vals = df.loc[df['v'] == v, 'num_mu_0'].unique()
    #     for num0 in mu_0_vals:
    #         ph = create_ph_arr_(num0, 2, mu_0, mu_1)
    #         marg_Bayes_prob_b[str(v) + '_' + str(num0)] = df.loc[
    #             (df['num_mu_0'] == num0) & (df['v'] == v), 'baysian_prob'].sum()
    #         t_1_probs = []
    #         for t_1 in range(t_shape-max_v):
    #             if np.sum(np.array(t_1_probs)) > 0.99999:
    #                 break
    #             curr_v = v
    #             curr_sum = 0
    #             for t in range(curr_v + 1, curr_v + 1 + t_1 + 1):
    #                 if t < t_prob.shape[0]:
    #                     if (t_prob[t] / np.sum(t_prob[curr_v + 1:])) > 0.000001:
    #                         curr_sum += (t_prob[t] / np.sum(t_prob[curr_v + 1:])) * \
    #                                     quad(create_t1_dens_, 0, 100, args=(t_1 - t + curr_v + 1, ph, lam_0, lam_1,))[0]
    #             t_1_probs.append(curr_sum)
    #         t1_dict_b[str(v) + '_' + str(num0)] = t_1_probs
    #
    # pkl.dump(t1_dict_b, open(t1_path_b, 'wb'))
    # pkl.dump(marg_Bayes_prob_b, open(bayes_prob_b, 'wb'))


def create_t1_dens_(r, h0, t_1, curr_ph, ph_before, ph_after, lam_0, lam_1):

    dens_1 = ph_dens(curr_ph, h0)
    dens_2 = ph_dens(ph_before, h0-r)
    dens_3 = ph_dens(ph_after, r)

    numin_1 = np.exp(-(lam_0 + lam_1) * r) * (((lam_0 + lam_1) * r) ** t_1)
    numin_2 = dens_2*dens_3

    denom_1 = math.factorial(t_1)
    denom_2 = dens_1

    return (numin_1 * numin_2) / (denom_1*denom_2)


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



def create_ph_before(num_mu0, num_lam0_1, num_mu0_lam0_1, lam_0, lam_1, mu_0):

    size = num_mu0 + num_lam0_1 + num_mu0_lam0_1

    ph = np.zeros((size, size))

    for ind in range(num_mu0):
        ph[ind, ind] = -mu_0
        if ind < size-1:
            ph[ind, ind + 1] = mu_0

    for ind in range(num_mu0, num_mu0 + num_lam0_1):
        ph[ind, ind] = -(lam_0 + lam_1)
        if ind < size - 1:
            ph[ind, ind + 1] = lam_0 + lam_1

    for ind in range(num_mu0 + num_lam0_1, num_mu0 + num_lam0_1 + num_mu0_lam0_1):
        ph[ind, ind] = -(mu_0 + lam_0 + lam_1)
        if ind < size - 1:
            ph[ind, ind + 1] = mu_0 + lam_0 + lam_1

    return ph


def create_ph_after(num_mu0, mu_0, mu_1):

    size = num_mu0 +1

    ph = np.zeros((size, size))

    for ind in range(num_mu0):
        ph[ind, ind] = -mu_0
        ph[ind, ind + 1] = mu_0

    ph[size-1, size-1] = -mu_1

    return ph


def ph_dens(ph, h):

    alph = np.zeros(ph.shape[0])
    alph[0] = 1
    ph_0 = -np.dot(ph, np.ones((ph.shape[0], 1)))

    return np.dot(alph, np.dot(expm(ph * h), ph_0))