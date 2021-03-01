import numpy as np
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import sympy
from sympy import *
from utils_ph import *
import matplotlib.pyplot as plt

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
        elif df.loc[ind,0] == 0:
            curr_prob = curr_prob * (steady_state[0]+steady_state[1])
        else:
            curr_prob = curr_prob*(steady_state[int(df.loc[ind, 0])+1])

        probs.append(curr_prob)

    return probs

def geometric_pdf(p,n):
    return p*((1-p)**(n))

def get_cdf_for_v(v, lam_0, lam_1, mu_0, mu_1, lam0, lam1, mu0, mu1, u0, u10, u11, R, x):
    pkl_name_inter_depart = '../pkl/combs' + str(v) + '.pkl'
    total_ph_lists = []
    with open(pkl_name_inter_depart, 'rb') as f:
        count, combp = pkl.load(f)


    df = pd.DataFrame(combp)
    for curr_ind in range(combp.shape[0]):
        comb = combp[curr_ind, :]
        ph = []

        if comb[1] == 1:
            ph.append('inter')
        for ph_ind in range(2, comb.shape[0]):
            if ph_ind % 2 == 0:
                if np.sum(comb[ph_ind:]) == 0:
                    ph.append(0)
                else:
                    # if ph_ind < comb.shape[0]-1:
                    if (comb[ph_ind] > 0) & (np.sum(comb[ph_ind + 1:]) == 0):
                        ph.append(comb[ph_ind])
                    elif (comb[ph_ind] == 0) & (np.sum(comb[ph_ind + 1:]) > 0):
                        ph.append(-1)
                    else:
                        curr_str = str(comb[ph_ind])
                        curr_str = curr_str + ',' + str(comb[ph_ind] + 1)
                        ph.append(curr_str)

            else:
                if comb[ph_ind] == int(1):
                    ph.append('inter')

        total_ph_lists.append(ph)

    df_list_path = '../pkl/df_list.pkl'
    with open(df_list_path, 'wb') as f:
        pkl.dump((df, total_ph_lists), f)

    a_list = []
    s_list = []
    for lis in total_ph_lists:
        # print(lis)
        a, s = create_ph_matrix_for_each_case(lis, lam0, lam1, mu0, mu1)
        a_list.append(a)
        s_list.append(s)

    steady_state = get_steady_for_given_v(u0, u10, u11, R, v)
    prob_for_each_case = compute_probs(df, total_ph_lists, steady_state, lam0, lam1, mu0, v)

    curr_cdf = 0

    for case_ind in range(len(a_list)):
        if type(prob_for_each_case[case_ind]) == sympy.core.mul.Mul:
            curr_prob = prob_for_each_case[case_ind].subs({lam0: lam_0, lam1: lam_1, mu0: mu_0})
        else:
            curr_prob = prob_for_each_case[case_ind]

        if np.array(s_list[case_ind].subs({lam0: lam_0, lam1: lam_1, mu0: mu_0})).shape[0] == 1:
            cdf = 1 - exp(np.array(s_list[case_ind].subs({lam0: lam_0, lam1: lam_1, mu0: mu_0}))[0][0] * x)
        else:
            cdf = 1 - np.sum(
                a_list[case_ind] * expm(np.array(s_list[case_ind].subs({lam0: lam_0, lam1: lam_1, mu0: mu_0})) * x))
        curr_cdf += cdf * curr_prob

    return curr_cdf

def main():

    with open(
            r'C:\Users\elira\PycharmProjects\redirected_git\redirected_call\inter_pkl\inter_deparature_distribution_service_03_08.pkl',
            'rb') as f:
        dff1 = pkl.load(f)

    dff1 = dff1.loc[3:, :]

    dff1_only_ones = dff1.loc[dff1['Class'] == 1, :].reset_index()
    for ind in range(dff1_only_ones.shape[0] - 1):
        dff1_only_ones.loc[ind + 1, 'inter_1'] = dff1_only_ones.loc[ind + 1, 'Time'] - dff1_only_ones.loc[ind, 'Time']



    lam_0 = 0.2
    lam_1 = 0.8
    mu_0 = 0.3
    mu_1 = 5000000.
    u0, u10, u11, R = get_steady(lam_0, lam_1, mu_0, mu_1)

    lam0, lam1, mu0, mu1 = symbols('lambda_0 lambda_1 mu_0 mu_1')


    probs = get_steady_for_given_v(u0, u10, u11, R, 2)

    p = lam_1 / (lam_1 + lam_0)

    emricial = []
    tot = dff1_only_ones.shape[0]
    theoretical = []
    x_vals = np.linspace(0.001, 20, 20)
    for x in tqdm(x_vals):
        total_pdf = (geometric_pdf(p,0))*(1-probs[0]-probs[1])+geometric_pdf(p,0)*(probs[0]+probs[1])*(1-np.exp(-(lam_0+lam_1)*x))
        for v in range(1, 6):
            curr_cdf = get_cdf_for_v(v, lam_0, lam_1, mu_0, mu_1, lam0, lam1, mu0, mu1, u0, u10, u11, R, x)
            # print(curr_cdf)
            total_pdf += curr_cdf*geometric_pdf(p,v)
        # print(total_pdf)
        theoretical.append(total_pdf)
        emricial.append(dff1_only_ones.loc[dff1_only_ones['inter_1'] < x, :].shape[0]/tot)




    linewidth = 5
    plt.figure()
    plt.plot(x_vals, np.array(emricial), alpha=0.7, linewidth=linewidth, label='Empirical', linestyle='dashed')
    plt.plot(x_vals, np.array(theoretical), alpha=0.7, linewidth=linewidth, label='Theoretical')
    plt.xlabel('X')
    plt.ylabel('CDF')
    plt.legend()
    plt.show()

    print('here')

if __name__ == '__main__':

    main()