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

    num_options_1 = np.array([1, 2, 2])
    options_list = [num_options_1]
    upper_bound = 16
    for num_options_ind in range(upper_bound):
        curr_array = np.array([1])
        for ind in range(1, options_list[num_options_ind].shape[0]):
            curr_array = np.append(curr_array, curr_array[ind - 1] + options_list[num_options_ind][ind])
        curr_array = np.append(curr_array, curr_array[-1])
        options_list.append(curr_array)



    mulam0_lam1_vals_list = []
    mulam0_lam1_vals_list.append(np.array([0, 1, 1, 1, 1]))

    lam0_lam1_vals_list = []
    lam0_lam1_vals_list.append(np.array([0, 0, 1, 1, 2]))

    mu_vals_list = []
    mu_vals_list.append(np.array([1, 1, 0, 1, 0]))

    lam0_lam1_vals_list_prob = []
    lam0_lam1_vals_list_prob.append(np.array([0, 1, 0, 1, 0]))

    mu_vals_list_prob = []
    mu_vals_list_prob.append(np.array([0, 0, 1, 0, 1]))

    mu_0 = 0.7
    lam_0 = 0.5
    lam_1 = 0.5
    mu_1 = 5000000.

    u0, u10, u11, R = get_steady(lam_0, lam_1, mu_0, mu_1)
    steady_arr = get_steady_for_given_v(u0, u10, u11, R, 1)



    data = np.concatenate((np.array(mu_vals_list).reshape(mu_vals_list[0].shape[0], 1).astype(int),
                           np.array(lam0_lam1_vals_list).reshape(mu_vals_list[0].shape[0], 1).astype(int),
                           np.array(mulam0_lam1_vals_list).reshape(mu_vals_list[0].shape[0], 1).astype(int),
                           np.array(mu_vals_list_prob).reshape(mu_vals_list[0].shape[0], 1).astype(int),
                           np.array(lam0_lam1_vals_list_prob).reshape(mu_vals_list[0].shape[0], 1).astype(int),
                           int(1 + 2)*(np.ones((mu_vals_list[0].shape[0], 1))),
                           np.array([steady_arr[-1], steady_arr[-2], steady_arr[-2], np.sum(steady_arr[:2]),
                                     np.sum(steady_arr[:2])]).reshape(mu_vals_list[0].shape[0], 1),
                           np.ones((mu_vals_list[0].shape[0], 1)),
                           geometric_pdf(lam_0, lam_1, 1)*np.ones((mu_vals_list[0].shape[0], 1))), axis=1)

    df = pd.DataFrame(data,
                      columns=['mu', 'lam0lam1', 'mu0lam0lam1', 'mu_prob', 'lam0lam1_prob', 'steady', 'steady_prob',
                               'v', 'v_prob'])

    df['mu'] = df['mu'].astype(int)
    df['lam0lam1'] = df['lam0lam1'].astype(int)
    df['mu0lam0lam1'] = df['mu0lam0lam1'].astype(int)

    df['mu_prob'] = df['mu_prob'].astype(int)
    df['lam0lam1_prob'] = df['lam0lam1_prob'].astype(int)

    df['prob_vent'] = df.mu_prob.apply(lambda x: (mu_0 / (mu_0 + lam_0 + lam_1)) ** x) * df.lam0lam1_prob.apply(
        lambda x: ((lam_0 + lam_1) / (mu_0 + lam_0 + lam_1)) ** x)
    df['total_prob'] = df['v_prob'] * df['steady_prob'] * df['prob_vent']

    df['event_ph'] = df.mu.apply(lambda x: str(x) + '_') + df.lam0lam1.apply(
        lambda x: str(x) + '_') + df.mu0lam0lam1.apply(lambda x: str(x))

    df_grp = df.groupby(['event_ph'])
    unique_vals = df['event_ph'].unique()
    event_arr = np.array([])
    prob_arr = np.array([])

    for event in unique_vals:
        event_arr = np.append(event_arr, event)
        prob_arr = np.append(prob_arr, df_grp.get_group(event)['total_prob'].sum())


    dat = np.concatenate((event_arr.reshape(event_arr.shape[0], 1), prob_arr.reshape(event_arr.shape[0], 1)), axis=1)


    df_acuum = pd.DataFrame(dat,  columns=['event', 'prob'])


    for v in range(2, upper_bound):

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


            elif ind < len(options_list[v - 1]) - 1:
                curr_assignment_mu = np.append(curr_new_mu[np.sum(options_list[v -1][:ind -1])
                                                        :np.sum(options_list[ v -1][:ind])],
                                            mu_vals_list[v - 2][np.sum(options_list[v - 2][:ind ])
                                                                :np.sum(options_list[v - 2][:ind + 1])])
                curr_new_mu = np.append(curr_new_mu, curr_assignment_mu)


                curr_assignment_lam0lam1 = np.append(curr_new_lam0lam1[np.sum(options_list[v - 1][:ind - 1])
                                                              :np.sum(options_list[v - 1][:ind])],
                                            lam0_lam1_vals_list[v - 2][np.sum(options_list[v - 2][:ind])
                                                                       :np.sum(options_list[v - 2][:ind + 1])])
                curr_new_lam0lam1 = np.append(curr_new_lam0lam1, curr_assignment_lam0lam1)


                curr_assignment_mu_lam0lam1 = np.append(curr_new_mulam0lam1[np.sum(options_list[v - 1][:ind - 1])
                                                                :np.sum(options_list[v - 1][:ind])],
                                            mulam0_lam1_vals_list[v - 2][np.sum(options_list[v - 2][:ind])
                                                                         :np.sum(options_list[v - 2][:ind + 1])])
                curr_new_mulam0lam1 = np.append(curr_new_mulam0lam1, curr_assignment_mu_lam0lam1 + 1)



                curr_assignment_mu_prob = np.append(curr_new_mu0_prob[np.sum(options_list[v - 1][:ind - 1])
                                                         :np.sum(options_list[v - 1][:ind])],
                                            mu_vals_list_prob[v - 2][np.sum(options_list[v - 2][:ind])
                                                                     :np.sum(options_list[v - 2][:ind + 1])] + 1)

                curr_new_mu0_prob = np.append(curr_new_mu0_prob, curr_assignment_mu_prob)


                curr_assignment_lam0lam1_prob = np.append(curr_new_lam0lam1_prob[np.sum(options_list[v - 1][:ind - 1])
                                                              :np.sum(options_list[v - 1][:ind])] + 1,
                                            lam0_lam1_vals_list_prob[v - 2][np.sum(options_list[v - 2][:ind])
                                                                            :np.sum(options_list[v - 2][:ind + 1])])
                curr_new_lam0lam1_prob = np.append(curr_new_lam0lam1_prob, curr_assignment_lam0lam1_prob)

            else:

                curr_new_mu = np.append(curr_new_mu, curr_assignment_mu)

                curr_assignment_lam0lam1 = curr_assignment_lam0lam1+1
                curr_new_lam0lam1 = np.append(curr_new_lam0lam1, curr_assignment_lam0lam1)

                curr_assignment_mu_lam0lam1 = curr_assignment_mu_lam0lam1+1
                curr_new_mulam0lam1 = np.append(curr_new_mulam0lam1, curr_assignment_mu_lam0lam1)

                curr_new_mu0_prob = np.append(curr_new_mu0_prob, curr_assignment_mu_prob)

                curr_new_lam0lam1_prob = np.append(curr_new_lam0lam1_prob, curr_assignment_lam0lam1_prob)

            if ind == 0:
                data = np.concatenate((curr_new_mu.reshape(1, 1).astype(int),
                                       curr_new_lam0lam1.reshape(1, 1).astype(int)
                                       , curr_new_mulam0lam1.reshape(1, 1).astype(int),
                                       curr_new_mu0_prob.reshape(1, 1).astype(int),
                                       curr_new_lam0lam1_prob.reshape(1, 1).astype(int),
                                      np.array([(int(v + 2))]).reshape(1,1),
                                      np.array([steady_arr[-1]]).reshape(1, 1),
                                      np.array(v).reshape(1, 1),
                                      np.array([geometric_pdf(lam_0, lam_1, v)]).reshape(1,1)),  axis=1)

            else:
                data = np.concatenate((curr_assignment_mu.reshape(curr_assignment_mu.shape[0], 1).astype(int),
                                       curr_assignment_lam0lam1.reshape(curr_assignment_lam0lam1.shape[0], 1).astype(int),
                                       curr_assignment_mu_lam0lam1.reshape(curr_assignment_mu_lam0lam1.shape[0], 1).astype(int),
                                       curr_assignment_mu_prob.reshape(curr_assignment_mu_prob.shape[0], 1).astype(int),
                                       curr_assignment_lam0lam1_prob.reshape(curr_assignment_lam0lam1_prob.shape[0], 1).astype(int),
                                       (v + 2 - ind)*np.ones((curr_assignment_mu.shape[0],1)),
                                       steady_arr[v + 2 - ind]*np.ones((curr_assignment_mu.shape[0],1)),
                                       v*np.ones((curr_assignment_mu.shape[0],1)),
                                       geometric_pdf(lam_0, lam_1, v)*np.ones((curr_assignment_mu.shape[0],1))), axis=1)

            df = pd.DataFrame(data, columns=['mu', 'lam0lam1', 'mu0lam0lam1', 'mu_prob', 'lam0lam1_prob', 'steady', 'steady_prob', 'v', 'v_prob'])

            df['mu'] = df['mu'].astype(int)
            df['lam0lam1'] = df['lam0lam1'].astype(int)
            df['mu0lam0lam1'] = df['mu0lam0lam1'].astype(int)

            df['mu_prob'] = df['mu_prob'].astype(int)
            df['lam0lam1_prob'] = df['lam0lam1_prob'].astype(int)

            df['prob_vent'] = df.mu_prob.apply(lambda x: (mu_0 / (mu_0 + lam_0 + lam_1)) ** x) * df.lam0lam1_prob.apply(
                lambda x: ((lam_0 + lam_1) / (mu_0 + lam_0 + lam_1)) ** x)
            df['total_prob'] = df['v_prob']*df['steady_prob']*df['prob_vent']

            df['event_ph'] = df.mu.apply(lambda x: str(x) + '_') + df.lam0lam1.apply(lambda x: str(x) + '_') + df.mu0lam0lam1.apply(lambda x: str(x))


            df_grp = df.groupby(['event_ph'])
            unique_vals = df['event_ph'].unique()
            event_arr = np.array([])
            prob_arr = np.array([])

            for event in unique_vals:
                event_arr = np.append(event_arr,event)
                prob_arr = np.append(prob_arr, df_grp.get_group(event)['total_prob'].sum())


            dat = np.concatenate((event_arr.reshape(event_arr.shape[0],1),prob_arr.reshape(event_arr.shape[0],1)), axis = 1)
            df_curr = pd.DataFrame(dat, columns=['event', 'prob'])
            # print(df_curr)




            # print('hi')
            df_acuum = pd.concat([df_curr, df_acuum])
            df_acuum['prob'] = df_acuum['prob'].astype(float)
            df_grp = df_acuum.groupby(['event'])
            unique_vals = df_acuum['event'].unique()
            event_arr = np.array([])
            prob_arr = np.array([])

            for event in unique_vals:
                event_arr = np.append(event_arr, event)
                prob_arr = np.append(prob_arr, df_grp.get_group(event)['prob'].sum())

            dat = np.concatenate((event_arr.reshape(event_arr.shape[0], 1), prob_arr.reshape(event_arr.shape[0], 1)),
                                 axis=1)

            df_acuum = pd.DataFrame(dat, columns=['event', 'prob'])


        mu_vals_list.append(curr_new_mu)
        lam0_lam1_vals_list.append(curr_new_lam0lam1)
        mulam0_lam1_vals_list.append(curr_new_mulam0lam1)

        mu_vals_list_prob.append(curr_new_mu0_prob)
        lam0_lam1_vals_list_prob.append(curr_new_lam0lam1_prob)


    print('stop')
    # print(curr_new_lam0lam1_prob)
    # print(curr_new_lam0lam1_prob)

if __name__ == '__main__':
    main()