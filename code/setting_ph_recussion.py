import numpy as np
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import sympy
from sympy import *
from utils_ph import *
import matplotlib.pyplot as plt
from utils_ph import create_ph_matrix_for_each_case, get_steady_for_given_v
import time
from numpy.linalg import matrix_power
import os
import scipy, math



def compute_probs(df, total_ph_lists, steady_state, lam_0, lam_1, mu_0, v):

    probs = [] # start a list of all the probabilities
    for ind in range(df.shape[0]): # going over each cave for the current v
        curr_prob = 1
        for curr_event in total_ph_lists[ind]:  # going over the current event

            if type(curr_event) != str:  # if the event is not a string
                if curr_event == 0:  # if the service is not conditioned no change in the probability
                    curr_prob = curr_prob*1
                elif curr_event > 0:  # if there are arrivals during the service
                    curr_prob = curr_prob*((lam_0+lam_1)/(lam_0+lam_1+mu_0))**curr_event
                else:
                    curr_prob = curr_prob * (mu_0 / (lam_0 + lam_1 + mu_0)) ** (-curr_event)  # the prob that the service
                    # is smaller than the inter_Arrival
            elif curr_event != 'inter': # there are two str events: inter and between service between lb and lb+1 arrivals
                vals = curr_event.split(',')
                lb = float(vals[0]) # it is basically a geometic distribution with mu_0/(lam_0 + lam_1 + mu_0)
                # and we wish to know the prob = lb+1
                curr_prob = curr_prob*((lam_0+lam_1) / ((lam_0 + lam_1 + mu_0)) ** lb)*(mu_0 / (lam_0 + lam_1 + mu_0))

        # each event is conditioned on a steady-state
        # the steady-state prob is determined by the number of customers left behind the first type 1 customer
        if df.loc[ind,0] == v+1:  #  this is a special case beucase the steady-state include P(U>=v+1)
            curr_prob = curr_prob*steady_state[-1]
        elif df.loc[ind,0] == 0:  # this can happen either if one or zero customers were left behind
            curr_prob = curr_prob * (steady_state[0]+steady_state[1])
        else:
            curr_prob = curr_prob*(steady_state[int(df.loc[ind, 0])+1])  # this is general case. The prob P(U = u+1)
            # is considered if u customers left behind

        probs.append(curr_prob)  # append the current prob

    return probs

def geometric_pdf(p,n):
    return p*((1-p)**(n))

def geometric_tail(p,n):
    return (1-p)**n

def get_ph_structure_for_v(v):

    start_time = time.time()
    # get the combination matrix
    print(v)
    pkl_name_inter_depart = '../pkl/combs' + str(v) + '.pkl'
    total_ph_lists = []
    with open(pkl_name_inter_depart, 'rb') as f:
        count, combp = pkl.load(f)

    # convert combination to pd dataframe
    df = pd.DataFrame(combp)
    for curr_ind in range(combp.shape[0]):  # go over each combination
        comb = combp[curr_ind, :]  # assign the current comb to 'comb'
        ph = []  # initiate a list of ph that convert the combination to its stochastic combination

        # constructing the ph combination
        if comb[1] == 1:  # this is an unusual case, if position 1 equals one then we start with an inter arrival
            ph.append('inter')
        for ph_ind in range(2, comb.shape[0]):  # go over the rest of the combinations
            if ph_ind % 2 == 0:  # if an even number
                if np.sum(comb[ph_ind:]) == 0:  # if there are no arrivals in the service and
                    # no more future arrival then regular service
                    ph.append(0)
                else:
                    if (comb[ph_ind] > 0) & (np.sum(comb[ph_ind + 1:]) == 0):  # if there are arrivals but no future arrivals
                        # then it is X|X> sum of y: from 1  to comb[ph_ind]
                        ph.append(comb[ph_ind])
                    elif (comb[ph_ind] == 0) & (np.sum(comb[ph_ind + 1:]) > 0): # if there are no arrivals in this service
                        # but there are future arrivals then X|X<Y
                        ph.append(-1)
                    else:  # this case reflects the case where there is a specific number of arrivals
                        curr_str = str(comb[ph_ind])
                        curr_str = curr_str + ',' + str(comb[ph_ind] + 1)
                        ph.append(curr_str)

            else:  # if it uneven position and the value is one it means we have an inter arrival
                if comb[ph_ind] == int(1):
                    ph.append('inter')

        total_ph_lists.append(ph)  # adding the current list to the list of the rest of the cases

    # dumping the list
    df_list_path = '../pkl/df_list'+ str(v) +'_.pkl'
    with open(df_list_path, 'wb') as f:
        pkl.dump((df, total_ph_lists), f)


    print("--- %s seconds for ph event construction with v=%d ---" % (time.time() - start_time, v))

def get_ph_representation(v, lam0, lam1, mu0, mu1):
    start_time = time.time()
    df_list_path = '../pkl/df_list'+ str(v) +'_.pkl'
    with open(df_list_path, 'rb') as f:
        df, total_ph_lists = pkl.load(f)


    # convert each list to its ph representation
    a_list = []
    s_list = []
    for lis in total_ph_lists:

        a, s = create_ph_matrix_for_each_case(lis, lam0, lam1, mu0, mu1)
        a_list.append(a)
        s_list.append(s)
    print("--- %s seconds for ph represenation construction with v=%d ---" % (time.time() - start_time, v))
    return a_list, s_list, total_ph_lists

def get_cdf(a_list, s_list, lam_0, lam_1, mu_0, x, prob_for_each_case, eps = 0.00001):

    curr_cdf = 0  # initiate with zero cdf
    not_included = 0
    cases_tracking = []
    for case_ind in range(len(a_list)):  #


        curr_prob = prob_for_each_case[case_ind]
        if curr_prob > eps:
            cdf = 1-np.sum(np.dot(a_list[case_ind], expm(s_list[case_ind]*x)))
            curr_cdf += cdf * curr_prob  # updating the current pdf
        else:
            not_included += curr_prob
            cases_tracking.append(case_ind)

    num_not_included = len(cases_tracking)
    if num_not_included > 0:
        middle_ind = int(num_not_included/2)

        first = (not_included/3)*(1-np.sum(np.dot(a_list[cases_tracking[0]], expm(s_list[cases_tracking[0]]*x))))
        middle = (not_included/3)*(1-np.sum(np.dot(a_list[cases_tracking[middle_ind]], expm(s_list[cases_tracking[middle_ind]]*x))))
        last = (not_included / 3) * (1 - np.sum(np.dot(a_list[cases_tracking[-1]], expm(s_list[cases_tracking[-1]] * x))))
        curr_cdf = curr_cdf + first+ middle + last
        # print(not_included)
    return curr_cdf


def get_pdf(a_list, s_list, lam0, lam1, mu0, lam_0, lam_1, mu_0, x, prob_for_each_case):
    '''

    :param a_list: the alpha (initial prob) for each case
    :param s_list: the generator matrix for each case
    :param lam0: type zero arrival rate - sympy
    :param lam1: type one arrival rate - sympy
    :param mu0: type zero arrival rate - sympy
    :param lam_0: type zero arrival rate - value
    :param lam_1: type one arrival rate - value
    :param mu_0: type zero arrival rate - value
    :param x: the current value of the pdf
    :param prob_for_each_case: a list of the prob for each case
    :return: the pdf value in x
    '''

    curr_pdf = 0

    for case_ind in range(len(a_list)):
        if type(prob_for_each_case[case_ind]) == sympy.core.mul.Mul:
            curr_prob = prob_for_each_case[case_ind].subs({lam0: lam_0, lam1: lam_1, mu0: mu_0})
        else:
            curr_prob = prob_for_each_case[case_ind]

        s_size = np.array(s_list[case_ind].subs({lam0: lam_0, lam1: lam_1, mu0: mu_0})).shape[0]
        curr_s0 = - np.dot(np.array(s_list[case_ind].subs({lam0: lam_0, lam1: lam_1, mu0: mu_0}))[0][0], np.ones((s_size, 1)))

        if np.array(s_list[case_ind].subs({lam0: lam_0, lam1: lam_1, mu0: mu_0})).shape[0] == 1:  # if scalar
            pdf = exp(np.array(s_list[case_ind].subs({lam0: lam_0, lam1: lam_1, mu0: mu_0}))[0][0] * x)*curr_s0

            pdf = pdf[0][0] # making it scalar
        else:
            pdf = a_list[case_ind] * expm(np.array(s_list[case_ind].subs({lam0: lam_0, lam1: lam_1, mu0: mu_0})) * x)*curr_s0
            pdf = pdf[0]

        curr_pdf += pdf * curr_prob

    return curr_pdf


def get_moment(a_list, s_list,  prob_for_each_case, moment = 1, eps = 0.0001):
    '''
    :param a_list: the alpha (initial prob) for each case
    :param s_list: the generator matrix for each case
    :param lam_0: type zero arrival rate - value
    :param lam_1: type one arrival rate - value
    :param mu_0: type zero arrival rate - value
    :param prob_for_each_case: a list of the prob for each case
    :return: moment of inter-departure times
    '''

    curr_mom = 0
    not_included = 0
    cases_tracking = []

    for case_ind in range(len(a_list)):

        curr_prob = prob_for_each_case[case_ind]

        if curr_prob > eps:

            mom = ((-1)**moment) * math.factorial(moment)*np.sum(np.dot(a_list[case_ind] , matrix_power((s_list[case_ind]), -moment)))

            curr_mom += mom * curr_prob
        else:
            not_included += curr_prob
            cases_tracking.append(case_ind)

    num_not_included = len(cases_tracking)
    if num_not_included > 0:
        middle_ind = int(num_not_included/2)

        first = (not_included/3)*((-1)**moment) * math.factorial(moment)*np.sum(np.dot(a_list[0] , matrix_power((s_list[0]), -moment)))
        middle = (not_included/3)*((-1)**moment) * math.factorial(moment)*np.sum(np.dot(a_list[middle_ind] , matrix_power((s_list[middle_ind]), -moment)))
        last = (not_included / 3) * ((-1)**moment) * math.factorial(moment)*np.sum(np.dot(a_list[-1] , matrix_power((s_list[-1]), -moment)))
        curr_mom = curr_mom + first+ middle + last

    return curr_mom


def main():

    with open(
            r'C:\Users\elira\PycharmProjects\redirected_git\redirected_call\inter_pkl\inter_deparature_distribution_service_07_lam1_05.pkl',
            'rb') as f:
        dff1 = pkl.load(f)

    dff1 = dff1.loc[3:, :]

    dff1_only_ones = dff1.loc[dff1['Class'] == 1, :].reset_index()
    for ind in range(dff1_only_ones.shape[0] - 1):
        dff1_only_ones.loc[ind + 1, 'inter_1'] = dff1_only_ones.loc[ind + 1, 'Time'] - dff1_only_ones.loc[ind, 'Time']

    main_path = '../pkl/'
    pkl_name = 'ph_cases_short'

    total_path = os.path.join(main_path, pkl_name)
    with open(total_path, 'rb') as f:
        ph_mat_v_list = pkl.load(f)

    pkl_name = 'df_short'
    total_path = os.path.join(main_path, pkl_name)
    with open(total_path, 'rb') as f:
        shrt_df = pkl.load(f)

    lam_0 = 0.5
    lam_1 = 0.5
    mu_0 = 0.7
    mu_1 = 5000000.
    u0, u10, u11, R = get_steady(lam_0, lam_1, mu_0, mu_1)

    probs = get_steady_for_given_v(u0, u10, u11, R, 2)

    p = lam_1 / (lam_1 + lam_0)

    start_time = time.time()
    v_low = 1
    v_high = 15

    emricial = []
    tot = dff1_only_ones.shape[0]
    theoretical = []
    x_vals = np.linspace(20, 0.01, 20)

    prob_atom = (geometric_pdf(p, 0))*(1-probs[0]-probs[1])
    print(prob_atom)

    if False:
        # cdf evaluation

        # The approximiation method
        with open('../pkl/ph_rep_approx_05.pkl', 'rb') as f:
            alpha, curr_T = pkl.load(f)

        approx_cdf = []
        time_tracker = []
        for x in tqdm(x_vals):
            approx_cdf.append(1 - np.sum(np.dot(alpha, expm(curr_T * x))))

            total_cdf = prob_atom+geometric_pdf(p,0)*(probs[0]+probs[1])*(1-np.exp(-(lam_0+lam_1)*x))

            start_time = time.time()

            for v in range(v_low, v_high):
                curr_cdf = 0

                for ind_ph in range(shrt_df[v-1].shape[0]):
                    alph = np.zeros(ph_mat_v_list[v-1][ind_ph].shape[0])
                    alph[0] = 1

                    curr_cdf += (1 - np.sum(np.dot(alph, expm(ph_mat_v_list[v-1][ind_ph] * x))) )* (
                                shrt_df[v-1].loc[ind_ph, 'prob'])



                    # print(curr_cdf)
                total_cdf += curr_cdf*geometric_pdf(p, v)
            time_tracker.append(time.time() - start_time)
            print("--- %s seconds for cdf x=%s ---" % (time.time() - start_time, x))

            theoretical.append(total_cdf)
            emricial.append(dff1_only_ones.loc[dff1_only_ones['inter_1'] < x, :].shape[0]/tot)

        print(time_tracker)

        linewidth = 3.5
        plt.figure()
        plt.plot(x_vals, np.array(emricial), alpha=0.8, linewidth=linewidth, label='Empirical', linestyle='dashed')
        plt.plot(x_vals, np.array(theoretical), alpha=0.7, linewidth=linewidth, label='Theoretical')
        plt.plot(x_vals,1-np.exp(-x_vals*lam_1), alpha=0.7, linewidth=linewidth, label='Exponential')
        plt.plot(x_vals, approx_cdf, alpha=0.7, linewidth=linewidth, label='Our approximation')
        plt.xlabel('X')
        plt.ylabel('CDF')
        plt.legend()
        plt.show()

        print('here')

    if True:





        time_tracker = []


        start_time = time.time()
        fifth_moment = prob_atom * 0 + geometric_pdf(p, 0) * (probs[0] + probs[1]) * (scipy.math.factorial(5) / (lam_0 + lam_1) ** 5)
        fourth_moment = prob_atom * 0 + geometric_pdf(p, 0) * (probs[0] + probs[1]) * (1 / (lam_0 + lam_1) ** 4)
        third_moment = prob_atom * 0 + geometric_pdf(p, 0) * (probs[0] + probs[1]) * (1 / (lam_0 + lam_1) ** 3)
        second_moment = prob_atom * 0 + geometric_pdf(p, 0) * (probs[0] + probs[1]) * (1 / (lam_0 + lam_1) ** 2)
        first_moment = prob_atom * 0 + geometric_pdf(p, 0) * (probs[0] + probs[1]) * (1 / (lam_0 + lam_1) ** 1)
        for v in range(v_low, v_high):

            for ind_ph in range(shrt_df[v-1].shape[0]):
                alph = np.zeros(ph_mat_v_list[v - 1][ind_ph].shape[0])
                alph[0] = 1

                fifth_moment += (- scipy.math.factorial(5) * np.sum(np.dot(alph, matrix_power(ph_mat_v_list[v-1][ind_ph],-5))) )* (
                            shrt_df[v-1].loc[ind_ph, 'prob'])*geometric_pdf(p, v)

                second_moment += (2 * np.sum(np.dot(alph, matrix_power(ph_mat_v_list[v-1][ind_ph],-2))) )* (
                            shrt_df[v-1].loc[ind_ph, 'prob'])*geometric_pdf(p, v)

                first_moment += (-np.sum(np.dot(alph, matrix_power(ph_mat_v_list[v-1][ind_ph],-1))) )* (
                            shrt_df[v-1].loc[ind_ph, 'prob'])*geometric_pdf(p, v)


        print(first_moment, second_moment,fifth_moment )

        time_tracker.append(time.time() - start_time)
        print("--- %s seconds for the  xth moment%s ---" % (time.time() - start_time, 5))




if __name__ == '__main__':

    main()