import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import os
from scipy.linalg import expm, sinm, cosm
import sys
sys.path.append(r'C:\Users\elira\Google Drive\butools2\Python')

from butools.ph import *
from butools.map import *
from butools.queues import *
from butools.mam import *
from butools.dph import *

from sympy import *
from numpy.linalg import matrix_power

from tqdm import tqdm
from utils_ph import *

def main():

    ## each v can be partitioned into different sets of c. here we compute the size of set
    num_options_1 = np.array([1, 2, 2])
    options_list = [num_options_1]
    upper_bound = 14
    for num_options_ind in range(upper_bound):
        curr_array = np.array([1])
        for ind in range(1, options_list[num_options_ind].shape[0]):
            curr_array = np.append(curr_array, curr_array[ind - 1] + options_list[num_options_ind][ind])
        curr_array = np.append(curr_array, curr_array[-1])
        options_list.append(curr_array)




    # number of phases for each rate for v=1

    mu_vals_list = np.array([1, 1, 0, 1, 0])
    lam0_lam1_vals_list = np.array([0, 0, 1, 1, 2])
    mulam0_lam1_vals_list = np.array([0, 1, 1, 1, 1])

    # number of the power for each probability
    mu_vals_list_prob = np.array([0, 0, 1, 0, 1])
    lam0_lam1_vals_list_prob = np.array([0, 1, 0, 1, 0])

    # assigning v = 1
    v = 1


    for ind in range(options_list[0].shape[0]): # dumping all the vectors in the recursion for v=1
        lower_bound = np.sum(options_list[v - 1][:ind])
        upper_bound_ = np.sum(options_list[v - 1][:ind+1])

        insert_to_pkl_v_1('mu', 'ph', lower_bound, upper_bound_, v, ind, mu_vals_list)
        insert_to_pkl_v_1('lam0lam1', 'ph', lower_bound, upper_bound_, v, ind, lam0_lam1_vals_list)
        insert_to_pkl_v_1('mulam0lam1', 'ph', lower_bound, upper_bound_, v, ind, mulam0_lam1_vals_list)
        insert_to_pkl_v_1('mu', 'prob', lower_bound, upper_bound_, v, ind, mu_vals_list_prob)
        insert_to_pkl_v_1('lam0lam1', 'prob', lower_bound, upper_bound_, v, ind, lam0_lam1_vals_list_prob)

    # system parameters
    mu_0 = 0.7
    lam_0 = 0.5
    lam_1 = 0.5
    mu_1 = 3.

    # we limit the size of updating the recursions
    units = 5

    # computing M/G/1 steady-state
    u0, u10, u11, R = get_steady(lam_0, lam_1, mu_0, mu_1)
    steady_arr = get_steady_for_given_v(u0, u10, u11, R, 1)



    # prob_b1 = get_prob_c(0, steady_arr, 2, mu_1, lam_0 + lam_1)
    # prob_b2 = get_prob_c(1, steady_arr, 2, mu_1, lam_0 + lam_1)
    # prob_b3 = get_prob_c(2, steady_arr, 2, mu_1, lam_0 + lam_1)
    # prob_b4 = get_prob_c(3, steady_arr, 2, mu_1, lam_0 + lam_1)
    #
    # summ = prob_b1+prob_b2+prob_b3+prob_b4


    #converting v=1 tuples and probabilites into a dataframe
    data = np.concatenate((np.array(mu_vals_list).reshape(mu_vals_list.shape[0], 1).astype(int),
                           np.array(lam0_lam1_vals_list).reshape(mu_vals_list.shape[0], 1).astype(int),
                           np.array(mulam0_lam1_vals_list).reshape(mu_vals_list.shape[0], 1).astype(int),
                           np.array(mu_vals_list_prob).reshape(mu_vals_list.shape[0], 1).astype(int),
                           np.array(lam0_lam1_vals_list_prob).reshape(mu_vals_list.shape[0], 1).astype(int),
                           int(1 + 2)*(np.ones((mu_vals_list.shape[0], 1))),
                           np.array([steady_arr[-1], steady_arr[-2], steady_arr[-2], np.sum(steady_arr[:2]),
                                     np.sum(steady_arr[:2])]).reshape(mu_vals_list.shape[0], 1),
                           np.ones((mu_vals_list.shape[0], 1)),
                           geometric_pdf(lam_0, lam_1, 1)*np.ones((mu_vals_list.shape[0], 1))), axis=1)

    # converting the data to pandas
    df = pd.DataFrame(data,
                      columns=['mu', 'lam0lam1', 'mu0lam0lam1', 'mu_prob', 'lam0lam1_prob', 'steady', 'steady_prob',
                               'v', 'v_prob'])

    # adding the marginal prob of events and a string of the event
    df = add_prob_even_total_prob(df, lam_0, lam_1, mu_0)

    # merging cases and leaving only the case and its final marginal prob

    df_acuum1 = merge_cases(df)
    df_acuum1['prob'] = df_acuum1['prob'].astype(float)

    # all the possible pairs we need to apply the recursion. Used later for dumping the pickles
    rate_ph_list = [('mu', 'ph'), ('lam0lam1', 'ph'), ('mulam0lam1', 'ph'), ('mu', 'prob'), ('lam0lam1', 'prob')]

    for v in tqdm(range(2, upper_bound)): # looping over all requirws values of v
        total_v_prob = 0  # tracking the total prob of each v (which should be 1), for debugging
        total_num_cases = 0  # tracking the number of cases in total, for debugging
        steady_arr = get_steady_for_given_v(u0, u10, u11, R, v) # steady state probs required

        # looping over the values of c for each v
        for ind, val in enumerate(options_list[v - 1]):

            if val < units:

                if ind == 0:  # dumping the recursion vectors for each pair in (ph_prob, val)
                    for rate_phprob in rate_ph_list:
                        insert_pkl_ind_0(rate_phprob[0], rate_phprob[1], v, ind)

                elif ind < len(options_list[v - 1]) - 1:  # dumping the recursion vectors for each pair in (ph_prob, val)

                    for rate_phprob in rate_ph_list:
                        insert_to_pkl_v_greater_2(rate_phprob[0], rate_phprob[1], v, ind, units)

                else: # dumping the recursion vectors for each pair in (ph_prob, val)

                    for rate_phprob in rate_ph_list:
                        insert_to_pkl_v_plus_1(rate_phprob[0], rate_phprob[1], v, ind, units)

                if ind == 0:  # we take all the curr recursion values and convert it into a df with event
                    # and marginal prob. also merge similar cases and sum their probabilities. Using pickles
                    a = []
                    for rate_phase in rate_ph_list:
                        curr_path = create_path_pkl(rate_phase[0], rate_phase[1], v, ind)
                        with open(curr_path, 'rb') as f:
                            curr_arr = pkl.load(f)[0]
                        a.append(curr_arr.reshape(1, 1).astype(int))
                    a.append(np.array([steady_arr[-1]]).reshape(1, 1))
                    a.append(np.array([geometric_pdf(lam_0, lam_1, v)]).reshape(1, 1))
                    data = np.concatenate((a[0], a[1], a[2], a[3], a[4], a[5], a[6]), axis=1)

                    df_curr1 = convert_to_pd_with_merge(data, lam_0, lam_1, mu_0)

                else:
                    a = []
                    for rate_phase in rate_ph_list:
                        curr_path = create_path_pkl(rate_phase[0], rate_phase[1], v, ind)
                        with open(curr_path, 'rb') as f:
                            curr_arr = pkl.load(f)[0]
                        a.append(curr_arr.reshape(curr_arr.shape[0], 1).astype(int))

                    if v + 2 - ind == 1:
                        a.append(np.sum(steady_arr[:2]) * np.ones((a[0].shape[0], 1)))
                    else:
                        a.append(steady_arr[v + 2 - ind]*np.ones((a[0].shape[0], 1)))

                    a.append(geometric_pdf(lam_0, lam_1, v)*np.ones((a[0].shape[0], 1)))

                    total_shape = a[0].shape[0]

                    data1 = np.concatenate((a[0], a[1], a[2], a[3], a[4], a[5], a[6]), axis=1)

                    df1 = pd.DataFrame(data1, columns=['mu', 'lam0lam1', 'mu0lam0lam1', 'mu_prob', 'lam0lam1_prob',
                                                     'steady_prob', 'v_prob'])
                    df1 = add_prob_even_total_prob(df1, lam_0, lam_1, mu_0)

                    df_curr1 = merge_cases(df1)
                    df_curr1['prob'] = df_curr1['prob'].astype(float)

                df_curr1['prob'] = df_curr1['prob'].astype(float)
                df_acuum1['prob'] = df_acuum1['prob'].astype(float)
                df_acuum1 = pd.concat([df_curr1, df_acuum1])
                df_acuum1['prob'] = df_acuum1['prob'].astype(float)

                df_acuum1 = merge_cases(df_acuum1)
                df_acuum1['prob'] = df_acuum1['prob'].astype(float)

            # total_v_prob += df_curr['prob'].sum()
            else:  # that is, there are more than units values

                if ind < len(options_list[v - 1]) - 1:  # dumping the recursion vectors for each pair in (ph_prob, val)

                    for rate_phprob in rate_ph_list:

                        insert_to_pkl_v_greater_2(rate_phprob[0], rate_phprob[1], v, ind, units)

                else:  # dumping the recursion vectors for each pair in (ph_prob, val)

                    for rate_phprob in rate_ph_list:
                        insert_to_pkl_v_plus_1(rate_phprob[0], rate_phprob[1], v, ind, units)

                if ind == 0:  # we take all the curr recursion values and convert it into a df with event
                    # and marginal prob. also merge similar cases and sum their probabilities. Using pickles
                    pass

                else:
                    a = []
                    for rate_phase in rate_ph_list:
                        curr_path = create_path_pkl(rate_phase[0], rate_phase[1], v, ind)
                        with open(curr_path, 'rb') as f:
                            curr_arr = pkl.load(f)[0]
                        a.append(curr_arr.reshape(curr_arr.shape[0], 1).astype(int))

                    if v + 2 - ind == 1:
                        a.append(np.sum(steady_arr[:2]) * np.ones((a[0].shape[0], 1)))
                    else:
                        a.append(steady_arr[v + 2 - ind] * np.ones((a[0].shape[0], 1)))

                    a.append(geometric_pdf(lam_0, lam_1, v) * np.ones((a[0].shape[0], 1)))

                    total_shape = a[0].shape[0]
                    if total_shape > units:

                        num_units = int(np.floor(total_shape / units))
                        for ind_units in tqdm(range(num_units + 1)):
                            if ind_units < num_units:

                                data1 = np.concatenate((a[0][ind_units * units: (ind_units + 1) * units],
                                                        a[1][ind_units * units: (ind_units + 1) * units],
                                                        a[2][ind_units * units: (ind_units + 1) * units],
                                                        a[3][ind_units * units: (ind_units + 1) * units],
                                                        a[4][ind_units * units: (ind_units + 1) * units],
                                                        a[5][ind_units * units: (ind_units + 1) * units],
                                                        a[6][ind_units * units: (ind_units + 1) * units]), axis=1)
                            else:

                                data1 = np.concatenate(
                                    (a[0][ind_units * units: ind_units * units + total_shape % units],
                                     a[1][ind_units * units: ind_units * units + total_shape % units],
                                     a[2][ind_units * units: ind_units * units + total_shape % units],
                                     a[3][ind_units * units: ind_units * units + total_shape % units],
                                     a[4][ind_units * units: ind_units * units + total_shape % units],
                                     a[5][ind_units * units: ind_units * units + total_shape % units],
                                     a[6][ind_units * units: ind_units * units + total_shape % units]),
                                    axis=1)

                            df1 = pd.DataFrame(data1,
                                               columns=['mu', 'lam0lam1', 'mu0lam0lam1', 'mu_prob', 'lam0lam1_prob',
                                                        'steady_prob', 'v_prob'])
                            df1 = add_prob_even_total_prob(df1, lam_0, lam_1, mu_0)

                            df1 = merge_cases(df1)
                            df1['prob'] = df1['prob'].astype(float)

                            if ind_units == 0:
                                df_total1 = df1
                            else:
                                df_total1 = pd.concat([df_total1, df1])

                        df_curr1 = merge_cases(df_total1)
                        df_curr1['prob'] = df_curr1['prob'].astype(float)

                    else:

                        data1 = np.concatenate((a[0], a[1], a[2], a[3], a[4], a[5], a[6]), axis=1)

                        df1 = pd.DataFrame(data1, columns=['mu', 'lam0lam1', 'mu0lam0lam1', 'mu_prob', 'lam0lam1_prob',
                                                           'steady_prob', 'v_prob'])
                        df1 = add_prob_even_total_prob(df1, lam_0, lam_1, mu_0)

                        df_curr1 = merge_cases(df1)
                        df_curr1['prob'] = df_curr1['prob'].astype(float)

                df_curr1['prob'] = df_curr1['prob'].astype(float)
                df_acuum1['prob'] = df_acuum1['prob'].astype(float)
                df_acuum1 = pd.concat([df_curr1, df_acuum1])
                df_acuum1['prob'] = df_acuum1['prob'].astype(float)

                df_acuum1 = merge_cases(df_acuum1)
                df_acuum1['prob'] = df_acuum1['prob'].astype(float)


        with open('../pkl/df_acuum.pkl', 'wb') as f:
            pkl.dump(df_acuum1, f)

        print(df_acuum1.shape[0])





def convert_to_pd_with_merge(data, lam_0, lam_1, mu_0):

    df = pd.DataFrame(data, columns=['mu', 'lam0lam1', 'mu0lam0lam1', 'mu_prob', 'lam0lam1_prob',
                                     'steady_prob', 'v_prob'])
    df = add_prob_even_total_prob(df, lam_0, lam_1, mu_0)

    df_curr = merge_cases(df)
    df_curr['prob'] = df_curr['prob'].astype(float)
    return df_curr

def insert_pkl_ind_0(rate, ph_prob, v, ind = 0):

    curr_full_path = create_path_pkl(rate, ph_prob, v, ind)
    if (rate == 'mu') & (ph_prob == 'ph'):
        arr = np.array([v])
    else:
        arr = np.array([0])

    with open(curr_full_path, 'wb') as f:
        pkl.dump((arr,  1), f)

def insert_to_pkl_v_greater_2(rate, ph_prob, v, ind, units ):

    pkl_path_1 = os.path.join('..\pkl', str(v), ph_prob, rate, str(ind-1))
    first_part_list = os.listdir(pkl_path_1)

    total_first_cases = 0
    for first_part_item in first_part_list:
        full_path = os.path.join(pkl_path_1, first_part_item)
        with open(full_path, 'rb') as f:
            curr_num = pkl.load(f)[1]
        total_first_cases += curr_num


    pkl_path_2 = os.path.join('..\pkl', str(v - 1), ph_prob, rate, str(ind))
    second_part_list = os.listdir(pkl_path_2)

    total_second_cases = 0
    for second_part_item in second_part_list:
        full_path = os.path.join(pkl_path_2, second_part_item)
        with open(full_path, 'rb') as f:
            curr_num = pkl.load(f)[1]
        total_second_cases += curr_num

    if total_first_cases + total_second_cases < units:

        first_part_path = create_path_pkl(rate, ph_prob, v, ind - 1)
        second_part_path = create_path_pkl(rate, ph_prob, v - 1, ind)

        with open(first_part_path, 'rb') as f:
            first_part = pkl.load(f)[0]
        with open(second_part_path, 'rb') as f:
            second_part = pkl.load(f)[0]

        if (ph_prob == 'ph') & (rate == 'mulam0lam1'):
            first_part += 1
            second_part += 1

        if (ph_prob == 'prob') & (rate == 'lam0lam1'):
            first_part += 1
        if (ph_prob == 'prob') & (rate == 'mu'):
            second_part += 1

    else:
        part = 1
        for curr_part in first_part_list:
            full_path = os.path.join(pkl_path_1, curr_part)
            with open(full_path, 'rb') as f:
                curr_vals = pkl.load(f)[0]
                if (ph_prob == 'ph') & (rate == 'mulam0lam1'):
                    curr_vals += 1
                if (ph_prob == 'prob') & (rate == 'lam0lam1'):
                    curr_vals += 1
            curr_path_pkl = create_path_pkl(rate, ph_prob, v, ind, part)
            with open(curr_path_pkl, 'wb') as f:
                pkl.dump((curr_vals, curr_vals.shape[0]), f)
            part += 1

        for curr_part in second_part_list:
            full_path = os.path.join(pkl_path_2, curr_part)
            with open(full_path, 'rb') as f:
                curr_vals = pkl.load(f)[0]
                if (ph_prob == 'ph') & (rate == 'mulam0lam1'):
                    curr_vals += 1
                if (ph_prob == 'prob') & (rate == 'mu'):
                    curr_vals += 1

            curr_path_pkl = create_path_pkl(rate, ph_prob, v, ind, part)
            with open(curr_path_pkl, 'wb') as f:
                pkl.dump((curr_vals, curr_vals.shape[0]), f)
            part += 1


    # total_shape = first_part.shape[0] + second_part.shape[0]
    # if (total_shape > units) & False:
    #     print('stop')
    #     num_units = int(np.floor(total_shape / units))
    #     for ind_units in tqdm(range(num_units + 1)):
    #         if ind_units < num_units:
    #             curr_path_pkl = create_path_pkl(rate, ph_prob, v, ind, ind_units+1)
    #
    #         else:
    #             pass
    #
    # else:
    #
    #     con_arr = np.append(first_part, second_part)
    #     # print('V=' + str(v) + ', ind = ' + str(ind) + 'rate: ' + rate + 'ph_prob: ' + ph_prob +  ' , arr = ' + str(con_arr))
    #
    #     curr_path_pkl = create_path_pkl(rate, ph_prob, v, ind)
    #     with open(curr_path_pkl, 'wb') as f:
    #         pkl.dump((con_arr, con_arr.shape[0]), f)

def insert_to_pkl_v_plus_1(rate, ph_prob, v, ind, units):

    pkl_path_1 = os.path.join('..\pkl', str(v), ph_prob, rate, str(ind - 1))
    first_part_list = os.listdir(pkl_path_1)

    # total_first_cases = 0
    # for first_part_item in first_part_list:
    #     full_path = os.path.join(pkl_path_1, first_part_item)
    #     with open(full_path, 'rb') as f:
    #         curr_num = pkl.load(f)[1]
    #     total_first_cases += curr_num

    part = 1
    for curr_part in first_part_list:
        full_path = os.path.join(pkl_path_1, curr_part)
        with open(full_path, 'rb') as f:
            curr_vals = pkl.load(f)[0]
            if (ph_prob == 'ph') & (rate == 'lam0lam1'):
                curr_vals += 1

        curr_path_pkl = create_path_pkl(rate, ph_prob, v, ind, part)
        with open(curr_path_pkl, 'wb') as f:
            pkl.dump((curr_vals, curr_vals.shape[0]), f)
        part += 1

    # first_part_path = create_path_pkl(rate, ph_prob, v, ind - 1)
    # with open(first_part_path, 'rb') as f:
    #     first_part = pkl.load(f)[0]
    #
    # if (ph_prob == 'ph') & (rate == 'lam0lam1'):
    #     first_part += 1
    #
    # con_arr = first_part
    # curr_path_pkl = create_path_pkl(rate, ph_prob, v, ind)
    # with open(curr_path_pkl, 'wb') as f:
    #     pkl.dump((con_arr, con_arr.shape[0]), f)

def insert_to_pkl_v_1(rate, ph_prob, lower_bound, upper_bound, v ,ind, val_arr):


    curr_full_path = create_path_pkl(rate, ph_prob, v, ind)

    arr = val_arr[lower_bound:upper_bound]

    with open(curr_full_path, 'wb') as f:
        pkl.dump((arr,  arr.shape[0]), f)


def create_path_pkl(rate, ph_prob, v, ind, part = 0):
    pkl_name = rate+'_' + ph_prob+'_' + str(v) + '_' + str(ind)
    pkl_path = os.path.join('..\pkl', str(v), ph_prob, rate, str(ind))
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path)
    if part > 0:
        pkl_name = pkl_name+ '_'+str(part)
    pkl_full_path = os.path.join(pkl_path, pkl_name)
    return pkl_full_path


def add_prob_even_total_prob(df,lam_0,lam_1,mu_0):
    '''

    :param df: initial dataframe
    :param lam_0: type 0 arrival rate
    :param lam_1: type 1 arrival rate
    :param mu_0: type 0 service rate
    :return: modified dataframe with marginal probabilites
    '''

    df['mu'] = df['mu'].astype(int)
    df['lam0lam1'] = df['lam0lam1'].astype(int)
    df['mu0lam0lam1'] = df['mu0lam0lam1'].astype(int)


    df['mu_prob'] = df['mu_prob'].astype(int)
    df['lam0lam1_prob'] = df['lam0lam1_prob'].astype(int)

    # computing the actual probability for each event
    df['prob_vent'] = df.mu_prob.apply(lambda x: (mu_0 / (mu_0 + lam_0 + lam_1)) ** x) * df.lam0lam1_prob.apply(
        lambda x: ((lam_0 + lam_1) / (mu_0 + lam_0 + lam_1)) ** x)
    # computing the total prob
    df['prob'] = df['v_prob'] * df['steady_prob'] * df['prob_vent']

    # represeting a case by the number of phases for each rate
    df['event'] = df.mu.apply(lambda x: str(x) + '_') + df.lam0lam1.apply(
        lambda x: str(x) + '_') + df.mu0lam0lam1.apply(lambda x: str(x))

    return df

def get_prob_c(c,steady_state,v, mu1, lam):
    '''

    :param c: number of customers to arrive in the future
    :param steady_state: M/G/1 steady-state
    :param v: num of type 0 customers
    :param mu1: type 1 arrival rate
    :param lam: conditioned arrival rate
    :return: a vector the probabilites of b for all states under (v,c)
    '''
    b = v+1-c
    prob_b = 0

    if b == 0:
        prob_b += np.sum(steady_state[:2])*prob_arrivals_during_exp(mu1, lam, 0)

    elif b == v+1:
        prob_b += np.sum(steady_state[:2])*tail_arrivals_during_exp(mu1, lam, v+1)
        u_arr = np.arange(2, b+1)
        k_arr = b-(u_arr-1)
        prob_b += np.sum(steady_state[u_arr]*tail_arrivals_during_exp(mu1, lam, k_arr))
        prob_b += steady_state[-1]
    else:
        prob_b += np.sum(steady_state[:2]) * prob_arrivals_during_exp(mu1, lam, b)
        u_arr = np.arange(2, b + 2)
        k_arr = b - (u_arr-1)
        prob_b += np.sum(steady_state[u_arr] * prob_arrivals_during_exp(mu1, lam, k_arr))

    return prob_b


def prob_arrivals_during_exp(mu1, lam, k):
    return (lam**k)*mu1/(lam+mu1)**(k+1)

def tail_arrivals_during_exp(mu1,lam, k):
    return (lam/(lam+mu1))**(k)

def merge_cases(df):
    df_grp = df.groupby(['event'])
    unique_vals = df['event'].unique()
    event_arr = np.array([])
    prob_arr = np.array([])

    for event in unique_vals:
        event_arr = np.append(event_arr, event)
        prob_arr = np.append(prob_arr, df_grp.get_group(event)['prob'].sum())

    dat = np.concatenate((event_arr.reshape(event_arr.shape[0], 1), prob_arr.reshape(event_arr.shape[0], 1)), axis=1)
    df_curr = pd.DataFrame(dat, columns=['event', 'prob'])
    return df_curr




if __name__ == '__main__':
    main()