import numpy as np
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import sympy
from sympy import *
from utils_ph import *

def compute_probs(df, total_ph_lists, steady_state, lam_0, lam_1, mu_0, v):

    probs = []
    for ind in range(df.shape[0]):
        curr_prob = 1
        for curr_event in total_ph_lists[ind]:

            if type(curr_event) != str:
                if curr_event == 0:
                    curr_prob = curr_prob*1
                elif curr_event > 0:
                    curr_prob = curr_prob*((lam_0+lam_1)/(lam_0+lam_1+mu_0))**curr_event
                else:
                    curr_prob = curr_prob * (mu_0 / (lam_0 + lam_1 + mu_0)) ** (-curr_event)
            elif curr_event != 'inter':

                vals = curr_event.split(',')
                lb = float(vals[0])
                curr_prob = curr_prob*((lam_0+lam_1) / ((lam_0 + lam_1 + mu_0)) ** lb)*(mu_0 / (lam_0 + lam_1 + mu_0))

        if df.loc[ind,0] == v+1:
            curr_prob = curr_prob*steady_state[-1]
        else:
            curr_prob = curr_prob*(steady_state[int(df.loc[ind, 0])+1])

        probs.append(curr_prob)

    return probs


def main():
    lam_0 = 0.1
    lam_1 = 0.9
    mu_0 = 0.2
    mu_1 = 5000000.

    u0, u10, u11, R = get_steady(lam_0, lam_1, mu_0, mu_1)

    lam0, lam1, mu_0 = symbols('lambda_0 lambda_1 mu_0')
    pkl_name_inter_depart = '../pkl/combs.pkl'
    total_ph_lists = []
    with open(pkl_name_inter_depart, 'rb') as f:
        count, combp = pkl.load(f)
    print(combp)
    v = 2
    df = pd.DataFrame(combp)
    for curr_ind in tqdm(range(combp.shape[0])):
        comb = combp[curr_ind,:]
        ph = []

        if comb[1] == 1:
            ph.append('inter')
        for ph_ind in range(2, comb.shape[0]):
            if ph_ind % 2 == 0:
                if np.sum(comb[ph_ind:]) == 0:
                    ph.append(0)
                else:
                    # if ph_ind < comb.shape[0]-1:
                    if (comb[ph_ind] > 0) & (np.sum(comb[ph_ind+1:]) == 0):
                        ph.append(comb[ph_ind])
                    elif (comb[ph_ind] == 0) & (np.sum(comb[ph_ind+1:]) > 0):
                        ph.append(-1)
                    else:
                        curr_str = str(comb[ph_ind])
                        curr_str = curr_str+',' + str(comb[ph_ind]+1)
                        ph.append(curr_str)

            else:
                if comb[ph_ind] == int(1):
                    ph.append('inter')

        total_ph_lists.append(ph)

    df_list_path  = '../pkl/df_list.pkl'
    with open(df_list_path, 'wb') as f:
        pkl.dump((df, total_ph_lists), f)



    for lis in total_ph_lists:

        print(lis)
    steady_state = get_steady_for_given_v(u0, u10, u11, R, v)
    prob_for_each_case = compute_probs(df, total_ph_lists, steady_state, lam0, lam1, mu_0, v)

    print('wait')


if __name__ == '__main__':

    main()