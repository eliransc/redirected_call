import pandas as pd
import pickle as pkl
import os
from utils_ph import *
import numpy as np
from numpy.linalg import matrix_power
from tqdm import tqdm
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import inv


def compute_ph_matrix(result, mu_0, mu_1, lam_0,lam_1, path_ph, ub_v, mean_num_rates_ub_v_path):



    result['ph_size'] = result['mu0'] + result['lam0lam1'] + result['lam0lam1mu0'] + 1

    eps = 10 ** (-3.9)
    if result.loc[result['prob'] < eps, 'mu0'].shape[0] > 0:
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

    mu0_avg_max_v, lam0lam1_avg_max_v, lam0lam1mu0_avg_max_v = pkl.load(open(mean_num_rates_ub_v_path, 'rb'))

    curr_row = result.shape[0]
    result.loc[curr_row, 'event'] = str(mu0_avg_max_v) + '_' + str(lam0lam1_avg_max_v) + '_' + str(lam0lam1mu0_avg_max_v)
    result.loc[curr_row, 'prob'] = epsilon
    result.loc[curr_row, 'mu0'] = mu0_avg_max_v
    result.loc[curr_row, 'lam0lam1'] = lam0lam1_avg_max_v
    result.loc[curr_row, 'lam0lam1mu0'] = lam0lam1mu0_avg_max_v
    result.loc[curr_row, 'ph_size'] = int(mu0_avg_max_v + lam0lam1_avg_max_v + lam0lam1mu0_avg_max_v + 1)



    ts = int(result['ph_size'].sum())
    print('The total ph size is: ', ts)
    probs = np.array(result['prob'])
    ph_size = np.array(result['ph_size']).astype(int)
    prob_arr = np.zeros((ts, 1))

    def create_erlang_row(rate, ind, size):
        aa = np.zeros(size)
        if ind <= size - 1:
            if ind == size - 1:
                aa[ind] = -rate
            else:
                aa[ind] = -rate
                aa[ind + 1] = rate
        return aa

    def generate_erlang_given_rates(rate, ph_size):

        A = np.identity(int(ph_size))
        A_list = [create_erlang_row(rate, ind, int(ph_size)) for ind in range(int(ph_size))]
        A = np.concatenate(A_list).reshape((int(ph_size), int(ph_size)))
        return A

    def create_gen_erlang_given_sizes(group_sizes, rates, probs=False):
        rates = rates[group_sizes > 0]
        group_sizes = group_sizes[group_sizes>0]

        ph_size = int(np.sum(group_sizes))
        erlang_list = [generate_erlang_given_rates(rates[ind], ph_size) for ind, ph_size in enumerate(group_sizes) if ph_size > 0 ]
        final_a = np.zeros((ph_size, ph_size))
        final_s = np.zeros(ph_size)
        if type(probs) == bool:
            rand_probs = np.random.dirichlet(np.random.rand(group_sizes.shape[0]), 1)
            rands = np.random.rand(group_sizes.shape[0])
            rand_probs = rands / np.sum(rands).reshape((1, rand_probs.shape[0]))
        else:
            rand_probs = probs
        for ind in range(group_sizes.shape[0]):
            final_s[int(np.sum(group_sizes[:ind]))] = rand_probs[0][ind]  # 1/diff_list.shape[0]
            final_a[int(np.sum(group_sizes[:ind])):int(np.sum(group_sizes[:ind]) + group_sizes[ind]),
            int(np.sum(group_sizes[:ind])):int(np.sum(group_sizes[:ind]) + group_sizes[ind])] = erlang_list[ind]

        for ind in range(group_sizes.shape[0]-1):
            final_a[int(np.sum(group_sizes[:ind+1]))-1,int(np.sum(group_sizes[:ind+1]))] = rates[ind]

        return final_s, final_a
    def extract_s_A(result, ind, mu_0, mu_1, lam_0,lam_1):
        s1, A1 = create_gen_erlang_given_sizes(
            np.array([result.loc[ind, 'mu0'], result.loc[ind, 'lam0lam1'], result.loc[ind, 'lam0lam1mu0'], 1]),
            np.array([mu_0, lam_0 + lam_1, lam_0 + lam_1 + mu_0, mu_1]), probs=False)

        s1 = np.zeros(int(result.loc[ind, 'ph_size']))
        s1[0] = 1

        return (s1, A1)



    def compute_lst(x, result, ind, mu_0, mu_1, lam_0,lam_1):

        prob_arr, ph = extract_s_A(result, ind, mu_0, mu_1, lam_0, lam_1)
        I = np.identity(ph.shape[0])
        inv_mat = inv(x * I - ph)
        first_lst = np.dot(prob_arr, inv_mat)
        ones = np.ones((ph.shape[0], 1))
        ph0 = -np.dot(ph, ones)
        lst = np.dot(first_lst, ph0)
        return lst


    # for x in np.linspace(0,10,40):
    #     vals = [probs[ind]*compute_lst(x, result, ind, mu_0, mu_1, lam_0,lam_1) for ind in range(result.shape[0])]
    #     print('The lst in {} is {}' .format(x, np.array(vals).flatten().sum()))


    initial_ind = []


    for ind_prob, prob in enumerate(probs):
        #     print(ind_prob, prob, np.sum(ph_size[:ind_prob]))
        prob_arr[np.sum(ph_size[:ind_prob])] = prob
        initial_ind.append(np.sum(ph_size[:ind_prob]))

    ph = np.zeros((ts, ts))

    def update_ph(init_ind, num_mu0, num_lam0lam1, num_lam0lam1mu0):
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

    import time
    start_time = time.time()

    ph1 = ph



    [update_ph(init_ind, int(result.loc[ind, 'mu0']), int(result.loc[ind, 'lam0lam1']), int(result.loc[ind, 'lam0lam1mu0'])) for ind, init_ind in enumerate(initial_ind)]

    print("--- %s seconds ---" % (time.time() - start_time))

    # start_time = time.time()
    # ph = np.zeros((ts, ts))
    #
    # for ind, init_ind in enumerate(initial_ind):
    #     num_mu0 = int(result.loc[ind, 'mu0'])
    #     num_lam0lam1 = int(result.loc[ind, 'lam0lam1'])
    #     num_lam0lam1mu0 = int(result.loc[ind, 'lam0lam1mu0'])
    #
    #
    #
    #
    #     for mu_0_ind in range(num_mu0):
    #         ph[init_ind + mu_0_ind, init_ind + mu_0_ind] = -mu_0
    #         ph[init_ind + mu_0_ind, init_ind + mu_0_ind + 1] = mu_0
    #
    #     for lam0lam1_ind in range(num_lam0lam1):
    #         ph[init_ind + num_mu0 + lam0lam1_ind, init_ind + num_mu0 + lam0lam1_ind] = -(lam_0 + lam_1)
    #         ph[init_ind + num_mu0 + lam0lam1_ind, init_ind + num_mu0 + lam0lam1_ind + 1] = lam_0 + lam_1
    #
    #     for lam0lam1mu0_ind in range(num_lam0lam1mu0):
    #         ph[init_ind + num_mu0 + num_lam0lam1 + lam0lam1mu0_ind, init_ind + num_mu0 + num_lam0lam1 + lam0lam1mu0_ind] \
    #             = -(lam_0 + lam_1 + mu_0)
    #         ph[init_ind + num_mu0 + num_lam0lam1 + lam0lam1mu0_ind, init_ind + num_mu0 + num_lam0lam1 + lam0lam1mu0_ind + 1] \
    #             = lam_0 + lam_1 + mu_0
    #
    #     ph[init_ind + num_mu0 + num_lam0lam1 + num_lam0lam1mu0, init_ind + num_mu0 + num_lam0lam1 + num_lam0lam1mu0] = -mu_1

    prob_arr = prob_arr.reshape((1, prob_arr.shape[0]))

    print("--- %s seconds ---" % (time.time() - start_time))


    size = ph.shape[0]
    I = np.identity(size)
    import time

    time_0 = time.time()

    # lst_list = []
    # for alph in tqdm(np.linspace(0,10,40)):
    #     inv_mat = inv(alph * I - ph)
    #     first_lst = np.dot(prob_arr, inv_mat)
    #     ones = np.ones((ph.shape[0], 1))
    #     ph0 = -np.dot(ph, ones)
    #     lst = np.dot(first_lst, ph0)
    #     lst_list.append(lst[0][0])

        # print('The lst in {} is {}'.format(alph, lst))
        #
        # print('A single lst derivation takes:', time.time()-time_0)

    # pkl.dump(lst_list, open('lst_list.pkl', 'wb'))
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(np.linspace(0,10,40),np.array(lst_list))
    # plt.show()

    PH_minus_3 = matrix_power(ph, -3)
    PH_minus_2 = matrix_power(ph, -2)
    PH_minus_1 = matrix_power(ph, -1)
    first_moment = -np.sum(np.dot(prob_arr, PH_minus_1))
    print(first_moment)
    second_moment = 2 * np.sum(np.dot(prob_arr, PH_minus_2))
    third_moment = -6 * np.sum(np.dot(prob_arr, PH_minus_3))
    variance = second_moment - (1 / lam_1) ** 2

    h_vals = []
    S0 = -np.dot(ph, np.ones((ph.shape[0], 1)))
    # for x in np.linspace(0, 3, 20):
    #     # curr_we = np.dot(np.dot(prob_arr, expm(x * ph)), S0)[0][0]
    #     curr_lst = np.dot(prob_arr, np.dot(matrix_power(x*np.identity(ph.shape[0])-ph, -1),S0))[0][0]
    #     h_vals.append(curr_lst)
    #     print(curr_lst)



    print('The true variance is: ', 1)
    print('The markovian variance is:', (1/lam_1)**2)

    pkl.dump((prob_arr, ph), open(path_ph, 'wb'))

    print((first_moment,second_moment, third_moment))

    return  (first_moment,second_moment, third_moment) #, h_vals
