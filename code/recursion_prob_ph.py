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
    mulam0_lam1_vals_list = np.array([0, 1, 1, 1, 1])

    lam0_lam1_vals_list= np.array([0, 0, 1, 1, 2])

    mu_vals_list = np.array([1, 1, 0, 1, 0])

    # number of the power for each probability
    lam0_lam1_vals_list_prob = np.array([0, 1, 0, 1, 0])

    mu_vals_list_prob = np.array([0, 0, 1, 0, 1])

    # system parameters
    mu_0 = 0.7
    lam_0 = 0.5
    lam_1 = 0.5
    mu_1 = 3.

    # we limit the size of updating the recursions
    units = 2000000

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

    df = pd.DataFrame(data,
                      columns=['mu', 'lam0lam1', 'mu0lam0lam1', 'mu_prob', 'lam0lam1_prob', 'steady', 'steady_prob',
                               'v', 'v_prob'])


    df = add_prob_even_total_prob(df, lam_0, lam_1, mu_0)

    df_acuum = merge_cases(df)
    df_acuum['prob'] = df_acuum['prob'].astype(float)


    for v in tqdm(range(2, upper_bound)):
        total_v_prob = 0
        total_num_cases = 0
        steady_arr = get_steady_for_given_v(u0, u10, u11, R, v)

        curr_new_mu = np.array([])
        curr_new_lam0lam1 = np.array([])
        curr_new_mulam0lam1 = np.array([])

        curr_new_mu0_prob = np.array([])
        curr_new_lam0lam1_prob = np.array([])


        for ind, val in enumerate(options_list[v - 1]):

            if ind == 0:
                curr_new_mu = np.append(curr_new_mu, v)
                curr_new_lam0lam1 = np.append(curr_new_lam0lam1, 0)
                curr_new_mulam0lam1 = np.append(curr_new_mulam0lam1, 0)

                curr_new_mu0_prob = np.append(curr_new_mu0_prob, 0)
                curr_new_lam0lam1_prob = np.append(curr_new_lam0lam1_prob, 0)
                total_num_cases += 1


            elif ind < len(options_list[v - 1]) - 1:
                curr_assignment_mu = np.append(curr_new_mu[np.sum(options_list[v -1][:ind -1])
                                                        :np.sum(options_list[v -1][:ind])],
                                            mu_vals_list[np.sum(options_list[v - 2][:ind ])
                                                                :np.sum(options_list[v - 2][:ind + 1])])
                curr_new_mu = np.append(curr_new_mu, curr_assignment_mu)


                curr_assignment_lam0lam1 = np.append(curr_new_lam0lam1[np.sum(options_list[v - 1][:ind - 1])
                                                              :np.sum(options_list[v - 1][:ind])],
                                            lam0_lam1_vals_list[np.sum(options_list[v - 2][:ind])
                                                                       :np.sum(options_list[v - 2][:ind + 1])])
                curr_new_lam0lam1 = np.append(curr_new_lam0lam1, curr_assignment_lam0lam1)


                curr_assignment_mu_lam0lam1 = np.append(curr_new_mulam0lam1[np.sum(options_list[v - 1][:ind - 1])
                                                                :np.sum(options_list[v - 1][:ind])],
                                            mulam0_lam1_vals_list[np.sum(options_list[v - 2][:ind])
                                                                         :np.sum(options_list[v - 2][:ind + 1])])

                curr_assignment_mu_lam0lam1 = curr_assignment_mu_lam0lam1 + 1
                curr_new_mulam0lam1 = np.append(curr_new_mulam0lam1, curr_assignment_mu_lam0lam1)



                curr_assignment_mu_prob = np.append(curr_new_mu0_prob[np.sum(options_list[v - 1][:ind - 1])
                                                         :np.sum(options_list[v - 1][:ind])],
                                            mu_vals_list_prob[np.sum(options_list[v - 2][:ind])
                                                                     :np.sum(options_list[v - 2][:ind + 1])] + 1)

                curr_new_mu0_prob = np.append(curr_new_mu0_prob, curr_assignment_mu_prob)


                curr_assignment_lam0lam1_prob = np.append(curr_new_lam0lam1_prob[np.sum(options_list[v - 1][:ind - 1])
                                                              :np.sum(options_list[v - 1][:ind])] + 1,
                                            lam0_lam1_vals_list_prob[np.sum(options_list[v - 2][:ind])
                                                                            :np.sum(options_list[v - 2][:ind + 1])])
                curr_new_lam0lam1_prob = np.append(curr_new_lam0lam1_prob, curr_assignment_lam0lam1_prob)

                total_num_cases += curr_assignment_lam0lam1_prob.shape[0]

            else:

                curr_new_mu = np.append(curr_new_mu, curr_assignment_mu)

                curr_assignment_lam0lam1 = curr_assignment_lam0lam1+1
                curr_new_lam0lam1 = np.append(curr_new_lam0lam1, curr_assignment_lam0lam1)

                curr_assignment_mu_lam0lam1 = curr_assignment_mu_lam0lam1
                curr_new_mulam0lam1 = np.append(curr_new_mulam0lam1, curr_assignment_mu_lam0lam1)

                curr_new_mu0_prob = np.append(curr_new_mu0_prob, curr_assignment_mu_prob)

                curr_new_lam0lam1_prob = np.append(curr_new_lam0lam1_prob, curr_assignment_lam0lam1_prob)

                total_num_cases += curr_assignment_lam0lam1_prob.shape[0]

            if ind == 0:
                data = np.concatenate((curr_new_mu.reshape(1, 1).astype(int),
                                       curr_new_lam0lam1.reshape(1, 1).astype(int)
                                       , curr_new_mulam0lam1.reshape(1, 1).astype(int),
                                       curr_new_mu0_prob.reshape(1, 1).astype(int),
                                       curr_new_lam0lam1_prob.reshape(1, 1).astype(int),
                                      # np.array([(int(v + 2))]).reshape(1,1),
                                      np.array([steady_arr[-1]]).reshape(1, 1),
                                      # np.array(v).reshape(1, 1),
                                      np.array([geometric_pdf(lam_0, lam_1, v)]).reshape(1,1)),  axis=1)

                df = pd.DataFrame(data, columns=['mu', 'lam0lam1', 'mu0lam0lam1', 'mu_prob', 'lam0lam1_prob',
                                                 'steady_prob', 'v_prob'])
                df = add_prob_even_total_prob(df, lam_0, lam_1, mu_0)

                df_curr = merge_cases(df)
                df_curr['prob'] = df_curr['prob'].astype(float)


            else:


                a1 = curr_assignment_mu.reshape(curr_assignment_mu.shape[0], 1).astype(int)
                a2 = curr_assignment_lam0lam1.reshape(curr_assignment_lam0lam1.shape[0], 1).astype(int)
                a3 = curr_assignment_mu_lam0lam1.reshape(curr_assignment_mu_lam0lam1.shape[0], 1).astype(int)
                a4 = curr_assignment_mu_prob.reshape(curr_assignment_mu_prob.shape[0], 1).astype(int)
                a5 = curr_assignment_lam0lam1_prob.reshape(curr_assignment_mu_prob.shape[0], 1).astype(int)

                if v + 2 - ind == 1:
                    a6 = np.sum(steady_arr[:2]) * np.ones((curr_assignment_mu.shape[0], 1))
                else:
                    a6 = steady_arr[v + 2 - ind]*np.ones((curr_assignment_mu.shape[0],1))
                a7 = geometric_pdf(lam_0, lam_1, v)*np.ones((curr_assignment_mu.shape[0],1))

                total_shape = a1.shape[0]
                if total_shape > units:

                    num_units = int(np.floor(total_shape/units))
                    for ind_units in tqdm(range(num_units + 1)):
                        if ind_units < num_units:
                            data = np.concatenate((a1[ind_units * units: (ind_units + 1) * units],
                                                   a2[ind_units * units: (ind_units + 1) * units],
                                                   a3[ind_units * units: (ind_units + 1) * units],
                                                   a4[ind_units * units: (ind_units + 1) * units],
                                                   a5[ind_units * units: (ind_units + 1) * units],
                                                   a6[ind_units * units: (ind_units + 1) * units],
                                                   a7[ind_units * units: (ind_units + 1) * units]), axis = 1)

                        else:
                            data = np.concatenate((a1[ind_units * units: ind_units * units + total_shape % units],
                                                   a2[ind_units * units: ind_units * units + total_shape % units],
                                                   a3[ind_units * units: ind_units * units + total_shape % units],
                                                   a4[ind_units * units: ind_units * units + total_shape % units],
                                                   a5[ind_units * units: ind_units * units + total_shape % units],
                                                   a6[ind_units * units: ind_units * units + total_shape % units],
                                                   a7[ind_units * units: ind_units * units + total_shape % units]), axis=1)

                        df = pd.DataFrame(data, columns=['mu', 'lam0lam1', 'mu0lam0lam1', 'mu_prob', 'lam0lam1_prob',
                                                         'steady_prob', 'v_prob'])
                        df = add_prob_even_total_prob(df, lam_0, lam_1, mu_0)


                        df = merge_cases(df)
                        df['prob'] = df['prob'].astype(float)


                        if ind_units == 0:
                            df_total = df
                        else:
                            df_total = pd.concat([df_total, df])

                    df_curr = merge_cases(df_total)
                    df_curr['prob'] = df_curr['prob'].astype(float)


                else:

                    data = np.concatenate((a1, a2, a3, a4, a5, a6, a7), axis=1)

                    df = pd.DataFrame(data, columns=['mu', 'lam0lam1', 'mu0lam0lam1', 'mu_prob', 'lam0lam1_prob',
                                                         'steady_prob', 'v_prob'])
                    df = add_prob_even_total_prob(df, lam_0, lam_1, mu_0)

                    df_curr = merge_cases(df)
                    df_curr['prob'] = df_curr['prob'].astype(float)


            df_curr['prob'] = df_curr['prob'].astype(float)
            df_acuum['prob'] = df_acuum['prob'].astype(float)
            df_acuum = pd.concat([df_curr, df_acuum])
            df_acuum['prob'] = df_acuum['prob'].astype(float)

            total_v_prob += df_curr['prob'].sum()


            df_acuum = merge_cases(df_acuum)
            df_acuum['prob'] = df_acuum['prob'].astype(float)


        with open('../pkl/df_acuum.pkl', 'wb') as f:
            pkl.dump(df_acuum, f)

        print(df_acuum.shape[0])


        mu_vals_list = curr_new_mu

        lam0_lam1_vals_list = curr_new_lam0lam1
        mulam0_lam1_vals_list = curr_new_mulam0lam1

        mu_vals_list_prob = curr_new_mu0_prob

        lam0_lam1_vals_list_prob = curr_new_lam0lam1_prob



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