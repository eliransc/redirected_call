import argparse
import sys
import numpy as np
import pickle as pkl
import os
import pandas as pd
from tqdm import tqdm
from num_cases_recusion import give_number_cases
from compute_df_probs_ph import compute_df
from tail_analytical import analytical_expression
from utils_ph import *
from create_ph_matrix import compute_ph_matrix
from get_steady_ph import get_steady_ph_sys
from compute_waiting_time import compute_waiting_time_
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import expm, sinm, cosm
import math
from tqdm import tqdm



def update_df(df, df_, t1, v, c, id, ar, init ,lam_0, lam_1, mu_0):
    curr_ind = df.shape[0]
    df.loc[curr_ind, 't1'] = t1
    df.loc[curr_ind, 'v'] = v
    df.loc[curr_ind, 'c'] = c
    df.loc[curr_ind, 'id'] = id
    df.loc[curr_ind, 'ar'] = ar
    df.loc[curr_ind, 'init'] = init
    df.loc[curr_ind, 'p_v'] = geometric_pdf(lam_0, lam_1, v)

    if c == 0:
        df.loc[curr_ind, 'p_a'] = 1
    else:
        l1 = ar-1
        l2 = c-id
        df.loc[curr_ind, 'p_a'] = ((mu_0/(mu_0+lam_0+lam_1))**l1)*(((lam_0+lam_1)/(mu_0+lam_0+lam_1))**l2)

    df.loc[curr_ind, 'n_v_c_id_ar'] = df_.loc[
        (df_['v'] == v) & (df_['c'] == c) & (df_['Id'] == id) & (df_['Ar'] == ar), 'number'].values[0]

    return df


def main(args):

    lam_0 = args.lam0
    lam_1 = args.lam1
    mu_0 = args.mu0
    mu_1 = args.mu1
    v_max = args.v_max

    pkl_path = r'../pkl'

    mean_num_rates_ub_v_path = os.path.join(pkl_path, str(args.v_max) + '_' + str(lam_0) + '_' + str(lam_1) + '_' + str(
        args.mu0) + '_' + str(args.mu1) + 'mean_nam_rate_ub_v.pkl')

    df_name_before = 'df_' + str(v_max) + '_' + str(lam_0) + '_' + str(lam_1) + '_' + str(mu_0) + '_' + str(
        mu_1) + '_before_probs.pkl'
    df_name_before = os.path.join(pkl_path, df_name_before)

    df_name_after = 'df_' + str(v_max) + '_' + str(lam_0) + '_' + str(lam_1) + '_' + str(mu_0) + '_' + str(
        mu_1) + '_after_probs.pkl'
    df_name_after = os.path.join(pkl_path, df_name_after)

    print('stage 1: compute general structure')
    if not os.path.exists(df_name_before):
        give_number_cases(v_max, df_name_before)

    df_ = pkl.load(open(df_name_before,'rb'))

    # print(df_)


    u0, u10, u11, R = get_steady(lam_0, lam_1, mu_0, mu_1)

    u_prob = [u0, u10+u11]
    for u in range(2, 2000):
        curr_prob = give_prob_u(u10, u11, R, u)

        u_prob.append(curr_prob)

        if curr_prob < args.eps:
            break
    t1_max = u
    u_prob = np.array(u_prob)
    # print(u)

    t1_prob = []
    # t1_max = 2000
    s = np.array([-mu_1]).reshape((1,1))
    for t1 in range(t1_max):
        curr_prob = 0
        for u in range(t1+1+1):
            if u == 0:
                t = t1
            else:
                t = t1-u+1
            curr_prob += u_prob[u]*quad(get_density, 0, 100, args=(s, lam_0, lam_1, t,))[0]\
                                                     / math.factorial(t)

        t1_prob.append(curr_prob)

        if curr_prob < args.eps:
            break

    t1_prob = np.array(t1_prob)



    columns = ['t1', 'v', 'c','id','ar', 'init', 'p_v','p_a','n_v_c_id_ar']
    df = pd.DataFrame([], columns = columns)


    # print(t1_prob)

    for t1 in range(t1_max):
        # t1 = 0
        # update v=0)
        v = 0
        b = min(v + 1, t1)
        c = v+1-b
        init = max(0, t1-(v + 1))
        if c == 0:
            df = update_df(df, df_, t1, v, c, 0, 0, init, lam_0, lam_1, mu_0)
        else:
            df = update_df(df, df_, t1, v, c, 1, 1, init, lam_0, lam_1, mu_0)

        # init = min(0, 0 + 1 - t1)
        #

        for v in range(1, args.v_max):
            b = min(v+1, t1)
            c = v + 1 - b
            init = max(0, t1 - (v + 1))
            if c == 0:
                id = ar = 0
                df = update_df(df, df_, t1, v, c, id, ar, init, lam_0, lam_1, mu_0)

            else:
                for id in range(c+1):
                    if (id == 0) & (c < v+1):
                        for ar in range(1, v+1):
                            df = update_df(df, df_, t1, v, c, id, ar, init ,lam_0, lam_1, mu_0)
                    elif (id > 0) & (c < v+1):
                        for ar in range(v+1-c+id, v+1+1):
                            df = update_df(df, df_, t1, v, c, id, ar, init, lam_0, lam_1, mu_0)
                    elif (id > 1) & (c == v+1):

                        for ar in range(v+1-c+id, v+1+1):
                            # print(v, c, id, ar)
                            df = update_df(df, df_, t1, v, c, id, ar, init, lam_0, lam_1, mu_0)
                    elif (id == 1) & (c == v+1):

                        for ar in range(v+1-c+id, v+1):
                            # print(v, c, id, ar)
                            df = update_df(df, df_, t1, v, c, id, ar, init, lam_0, lam_1, mu_0)

    df['total_prob'] = df['p_a'] * df['p_v'] * df['n_v_c_id_ar']
    df['num_mu0'] = df.apply(lambda x: num_mu0(x.v, x.ar), axis=1)
    df['num_mu1'] = df.apply(lambda x: num_mu1(x.ar), axis=1)

    t2_arr = np.zeros(args.max_t2)

    for ind in tqdm(range(df.shape[0])):
        size = df.loc[ind, 'num_mu0'] + df.loc[ind, 'num_mu1']
        curr_arr = np.zeros((size,size))
        for ind_arr in range(df.loc[ind, 'num_mu1']):
            curr_arr[ind_arr, ind_arr] = -mu_1
            if (size > 1) & (df.loc[ind, 'num_mu0'] >0):
                curr_arr[ind_arr, ind_arr + 1] = mu_1

        for ind_arr in range(df.loc[ind, 'num_mu1'], size):
            curr_arr[ind_arr, ind_arr] = -mu_0
            if ind_arr < size-1:
                curr_arr[ind_arr, ind_arr + 1] = mu_0

        for t2 in range(t2_arr.shape[0]):

            if t2 >= df.loc[ind, 'init']:
                t2_ = t2-df.loc[ind, 'init']
                df.loc[ind, 't_2_given_t1_'+str(t2)] = quad(get_density, 0, 100, args=(curr_arr, lam_0, lam_1, t2_,))[0]\
                                                     / math.factorial(t2_)

    df['t1_prob'] = df.apply(lambda x: comp_t1(t1_prob, x.t1), axis=1)

    for t2_val in range(args.max_t2):

        df['total_prob_t2_given_t1_'+str(t2_val)] = df['total_prob'] * df['t_2_given_t1_'+str(t2_val)]

    pkl.dump(df, open('df_cond.pkl', 'wb'))
    pkl.dump(t1_prob, open('t1_prob.pkl', 'wb'))


    print('fff')


def get_density(h,  S, lam0, lam1,  k):

    S0 = -np.dot(S, np.ones((S.shape[0], 1)))
    alph = np.zeros(S.shape[0])
    alph[0] = 1

    return np.exp(-(lam0 + lam1) * h) * (((lam0 + lam1) * h) ** k) * np.dot(np.dot(alph, expm(S * h)), S0)

def comp_t1(t1_prob, t1):
    return t1_prob[t1]

def num_mu0(v, ar):

    if v == 0:
        return 0
    else:
        return v-ar+1

def num_mu1(ar):

    if ar == 0:
        return 2
    else:
        return 1


def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--v_max', type=int, help='v_max', default=7)
    parser.add_argument('--max_t2', type=int, help='max_t2 we compute', default=15)
    parser.add_argument('--mu0', type=float, help='mu0', default=0.25)
    parser.add_argument('--mu1', type=float, help='mu1', default=25)
    parser.add_argument('--lam0', type=float, help='mu0', default=0.1)
    parser.add_argument('--lam1', type=float, help='mu0', default=0.9)
    parser.add_argument('--lam_ext', type=float, help='external arrival to sub queue', default=0.5)
    parser.add_argument('--mu_11', type=float, help='service rate in sub queue', default=1.5)
    parser.add_argument('--eps', type=float, help='mu0', default=0.0000001)



    args = parser.parse_args(argv)

    return args

if __name__ =='__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)