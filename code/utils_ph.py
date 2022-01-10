import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import *
import sys
sys.path.append(r'G:\My Drive\butools2\Python')
sys.path.append('/home/d/dkrass/eliransc/Python')
import os
from tqdm import tqdm
from butools.ph import *
from butools.map import *
from butools.queues import *
from butools.mam import *
from butools.dph import *
from scipy.linalg import expm, sinm, cosm
from sympy import *
from sympy import Symbol
from sympy.physics.quantum import TensorProduct
import pickle as pkl
import pandas as pd
from sympy import diff, sin, exp
from numpy.linalg import matrix_power
import time


def busy(s, lam2, mu2):
    return ((lam2 + mu2 + s) - ((lam2 + mu2 + s) ** 2 - 4 * lam2 * mu2) ** 0.5) / (2 * lam2)

def prob_arrivals_during_exp(mu1, lam, k):
    return (lam**k)*mu1/(lam+mu1)**(k+1)

def tail_arrivals_during_exp(mu1,lam, k):
    return (lam/(lam+mu1))**(k)

def ser_lap(s, mu):
    return mu / (s + mu)


def hyper(s, lam1, lam2, mu1, mu2):
    return ser_lap(s, mu1) * lam1 / (lam1 + lam2) + ser_lap(s, mu2) * lam2 / (lam1 + lam2)


def rho(lam1, lam2, mu1, mu2):
    return (lam1 + lam2) * ((lam1 / ((lam1 + lam2) * mu1)) + (lam2 / ((lam1 + lam2) * mu2)))


def w_lap(s, lam1, lam2, mu1, mu2):
    return ((1 - rho(lam1, lam2, mu1, mu2)) * s) / (s - (lam1 + lam2) * (1 - hyper(s, lam1, lam2, mu1, mu2)))


def F(s, lam1, lam2, mu1, mu2):
    return w_lap(s, lam1, lam2, mu1, mu2) * ser_lap(s, mu1)


def A(s, lam1, lam2, mu2):
    return (lam1 / (lam1 + lam2 - lam2 * (ser_lap(s, mu2))))


def beta(s, lam1, lam2, mu1, mu2):
    return (lam1 / (lam1 + lam2 + s) + ((A(s, lam1, lam2, mu2) * lam2) / (lam1 + lam2 + s)) * (
                ser_lap(s, mu2) - busy(s + lam1, lam2, mu2))) / (
                       1 - ((lam2 * busy(s + lam1, lam2, mu2)) / (lam1 + lam2 + s)))


def tau(s, lam1, lam2, mu1, mu2):
    return ser_lap(s, mu1) * (A(s, lam1, lam2, mu2) * (
                1 - F(lam1 + lam2 - lam2 * busy(s + lam1, lam2, mu2), lam1, lam2, mu1, mu2)) + F(
        lam1 + lam2 - lam2 * busy(s + lam1, lam2, mu2), lam1, lam2, mu1, mu2) * beta(s, lam1, lam2, mu1, mu2))


def get_var(lam1, lam2, mu1, mu2):
    s = Symbol('s')
    y = tau(s, lam1, lam2, mu1, mu2)
    dx = diff(y, s)
    dxdx = diff(dx, s)
    return dxdx.subs(s, 0) - (dx.subs(s, 0)) ** 2


def get_nth_moment(lam1, lam2, mu1, mu2, n):
    s = Symbol('s')
    y = tau(s, lam1, lam2, mu1, mu2)
    for i in range(n):
        if i == 0:
            dx = diff(y, s)
        else:
            dx = diff(dx, s)
    return dx.subs(s, 0)


def get_first_n_moments(parameters, n=5):
    lam1, lam2, mu1, mu2 = parameters
    moments = []
    for n in range(1, n + 1):
        moments.append(get_nth_moment(lam1, lam2, mu1, mu2, n) * (-1) ** n)
    moments = np.array([moments], dtype='float')
    return moments


def kroneker_sum(G, H):
    size_g = G.shape[0]
    size_h = H.shape[0]
    return np.kron(G, np.identity(size_h)) + np.kron(np.identity(size_g), H)


def give_boundry_probs(R, A0, A1, A, B, C0, ro):
    p00, p01, p02, p100, p110, p120, p101, p111, p121 = symbols('p00 p01 p02 p100 p110 p120 p101 p111 p121')
    eqns = [np.dot(np.array([p00, p01, p02]), np.ones((A0.shape[0]))) - (1 - ro)]
    eq3 = np.dot(np.array([p00, p01, p02]), A0) + np.dot(np.array([p100, p110, p120, p101, p111, p121]), A1)
    eq1 = np.dot(np.array([p00, p01, p02]), C0)
    eq2 = np.dot(np.array([p100, p110, p120, p101, p111, p121]), B + np.dot(R, A))
    for eq_ind in range(B.shape[0]):
        eqns.append(eq1[0, eq_ind] + eq2[0, eq_ind])

    for eq_ind in range(A0.shape[0]):
        eqns.append(eq3[0, eq_ind])

    A_mat, b = linear_eq_to_matrix(eqns[:-1], [p00, p01, p02, p100, p110, p120, p101, p111, p121])
    return A_mat, b


def get_expect_gph_system(R, p1_arr, xm_max=5000):
    expected = 0
    for pi_val in range(1, xm_max):
        ui = p1_arr.reshape((1, R.shape[0]))
        Ri = np.linalg.matrix_power(R, pi_val - 1)
        expected += np.dot(np.dot(ui, Ri), np.ones((R.shape[0], 1))) * pi_val
    return expected[0, 0]


def get_expect_gph_system(R, p1_arr, xm_max=5000):
    expected = 0
    for pi_val in range(1, xm_max):
        ui = p1_arr.reshape((1, R.shape[0]))
        Ri = np.linalg.matrix_power(R, pi_val - 1)
        expected += np.dot(np.dot(ui, Ri), np.ones((R.shape[0], 1))) * pi_val
    return expected[0, 0]


def get_A0(Ts):
    krom_sum = kroneker_sum(Ts[0], Ts[1])
    if len(Ts) > 2:
        for T_ind in range(2, len(Ts)):
            krom_sum = kroneker_sum(krom_sum, Ts[T_ind])
    return krom_sum


def get_C_first(T0s, Ts, s):
    krom_sum = kroneker_sum(T0s[0], T0s[1])
    if len(Ts) > 2:
        for T_ind in range(2, len(Ts)):
            krom_sum = kroneker_sum(krom_sum, T0s[T_ind])
    return krom_sum


def get_B(Ts, s):
    krom_sum = kroneker_sum(Ts[0], Ts[1])
    if len(Ts) > 2:
        for T_ind in range(2, len(Ts)):
            krom_sum = kroneker_sum(krom_sum, Ts[T_ind])
    return kroneker_sum(krom_sum, s)


def get_A(Ts, new_beta, s0):
    kron_sum = kroneker_sum(np.zeros(Ts[0].shape[0]), np.zeros(Ts[1].shape[0]))

    if len(Ts) > 2:
        for T_ind in range(2, len(Ts)):
            kron_sum = kroneker_sum(kron_sum, np.zeros(Ts[T_ind].shape[0]))
    kron_sum = kroneker_sum(kron_sum, np.dot(s0, new_beta))
    return kron_sum


def compute_s_beta(r, mu, num_stations=2):
    s_ = np.array([])
    total_arrivals_to_station = np.sum(r[:, station_ind]) + np.sum(r[station_ind, :]) - np.sum(
        r[station_ind, station_ind])
    beta = np.array([])
    for stream_ind in range(r.shape[0]):
        if r[station_ind, stream_ind] > 0:
            beta = np.append(beta, r[station_ind, stream_ind] / total_arrivals_to_station)
            s_ = np.append(s_, -mu[station_ind, stream_ind])
    for out_station in range(num_stations):
        if out_station != station_ind:
            if r[out_station, station_ind] > 0:
                beta = np.append(beta, r[out_station, station_ind] / total_arrivals_to_station)
                s_ = np.append(s_, -mu[station_ind, station_ind])
    new_beta = np.array([])
    new_s_ = np.unique(s_)
    for val in new_s_:
        new_beta = np.append(new_beta, np.sum(beta[np.argwhere(s_ == val)]))
    new_beta = new_beta.reshape((1, new_beta.shape[0]))
    s = np.identity(new_s_.shape[0]) * new_s_

    return s, new_beta, new_s_


def compute_curr_t(curr_ind, r, mu):
    r_mismatched = np.sum(r[curr_ind, :]) - r[curr_ind, curr_ind]
    r_matched = r[curr_ind, curr_ind]
    mu_mismatched = np.mean(np.delete(mu[curr_ind, :], curr_ind, 0))
    mu_matched = mu[curr_ind, curr_ind]
    parameters = (r_mismatched, r_matched, mu_mismatched, mu_matched)

    moments = get_first_n_moments(parameters)
    return moments


def give_tail_analytic(num_mu0, numlam0lam1, nummu0lam0lam1, mu_0, lam_0, lam_1, mu_1,x):
    num_mu0 = int(num_mu0)
    numlam0lam1 = int(numlam0lam1)
    nummu0lam0lam1 = int(nummu0lam0lam1)

    size = num_mu0 + numlam0lam1 + nummu0lam0lam1 + 1

    limit1 = num_mu0
    limit2 = num_mu0 + numlam0lam1
    limit3 = size - 1

    s = np.zeros((size, size))
    for ind in range(s.shape[0]):
        if ind < limit1:
            s[ind, ind] = -mu_0
            s[ind, ind + 1] = mu_0
        elif ind < limit2:
            s[ind, ind] = -(lam_0 + lam_1)
            s[ind, ind + 1] = lam_0 + lam_1
        elif ind < limit3:
            s[ind, ind] = -(lam_0 + lam_1 + mu_0)
            s[ind, ind + 1] = lam_0 + lam_1 + mu_0
        else:
            s[ind, ind] = -mu_1

    alpha = np.zeros(size)
    alpha[0] = 1

    return Matrix(alpha).transpose() * (exp(Matrix(s) * x)) * ones(num_mu0 + numlam0lam1 + nummu0lam0lam1 + 1, 1)


def get_Ts_alphas(r, mu, station_ind):
    alphas = []
    Ts = []
    T0s = []

    for curr_ind in range(r.shape[0]):
        if curr_ind != station_ind:
            mome = compute_curr_t(curr_ind, r, mu)
            curr_alpha, curr_T = PH3From5Moments(mome[0])
            alphas.append(curr_alpha)
            Ts.append(curr_T)
            T0s.append(-np.dot(np.dot(curr_T, np.ones((curr_T.shape[0], 1))), curr_alpha))
    for stream_ind in range(r[station_ind, :].shape[0]):
        Ts.append(np.array(-r[station_ind, stream_ind]).reshape((1, 1)))
        alphas.append(1.)
        T0s.append(-np.dot(np.dot(Ts[-1], np.ones(1)), alphas[-1]))

    return Ts, T0s, alphas


def total_arrivals_to_station(r):
    return np.sum(r[:, station_ind]) + np.sum(r[station_ind, :]) - np.sum(r[station_ind, station_ind])


def get_ro(r, mu, new_beta, new_s_):
    return np.sum(new_beta * total_arrivals_to_station(r) * (-1 / new_s_))


def get_ro_2(lam_0, lam_1, new_beta, s0):
    return (lam_0 + lam_1) * np.dot(new_beta, 1 / s0)


from numpy.linalg import matrix_power


def get_bound_steady_state(R, A0, A1, AA, B, C0, ro):
    u0, u10, u11 = symbols('u0 u10  u11')
    eqns = [u0 - (1 - ro[0][0])]
    for ind in range(2):
        eqns.append(np.dot(u0, C0)[0][ind] + np.dot(np.array([u10, u11]), B)[ind] +
                    np.dot(np.dot(np.array([u10, u11]), R), AA)[0][0, ind])
    A_mat, b = linear_eq_to_matrix(eqns, [u0, u10, u11])
    u0, u10, u11 = np.linalg.solve(np.array(A_mat, dtype=np.float), np.array(b, dtype=np.float))
    return u0[0], u10[0], u11[0]

def create_curr_ph_inter(num_mu0, num_lam0_1, num_mu0_lam0_1,  lam_0, lam_1, mu_0, mu_1):



    size = num_mu0 + num_lam0_1 + num_mu0_lam0_1 + 1

    ph = np.zeros((size, size))

    for ind in range(num_mu0):
        ph[ind, ind] = -mu_0
        ph[ind, ind + 1] = mu_0

    for ind in range(num_mu0, num_mu0 + num_lam0_1):
        ph[ind, ind] = -(lam_0 + lam_1)
        ph[ind, ind + 1] = lam_0 + lam_1

    for ind in range(num_mu0 + num_lam0_1, num_mu0 + num_lam0_1 + num_mu0_lam0_1):
        ph[ind, ind] = -(mu_0 + lam_0 + lam_1)
        ph[ind, ind + 1] = mu_0 + lam_0 + lam_1

    ph[size - 1, size - 1] = -mu_1

    return ph

def get_Avg_system(R, u10, u11):
    p1 = np.array([u10, u11])
    total_avg = 0
    for ind in range(1, 500):
        total_avg += ind * np.sum(np.dot(p1, matrix_power(R, ind - 1)))
    return total_avg


def get_steady(lam_0, lam_1, mu_0, mu_1):

    T0 = np.array([-lam_0])
    T1 = np.array([-lam_1])
    Ts = [T0, T1]
    T00 = np.array([-np.dot(T0, np.ones(1))])
    T10 = np.array([-np.dot(T1, np.ones(1))])

    T0s = [T00, T10]
    alphas = [np.array(1.), np.array(1.), ]

    new_beta = np.array([lam_0 / (lam_0 + lam_1), lam_1 / (lam_0 + lam_1)]).reshape(1, 2)
    s = np.array([[-mu_0, 0], [0, -mu_1]])
    s0 = -np.dot(s, np.ones((s.shape[0], 1)))

    s0 = -np.dot(s, np.ones((s.shape[0], 1)))
    A0 = get_A0(Ts)
    A1 = np.kron(np.identity(A0.shape[0]), s0)
    AA = get_A(Ts, new_beta, s0)
    B = get_B(Ts, s)
    C = kroneker_sum(get_C_first(T0s, Ts, s), np.zeros(s.shape))
    C0 = np.kron(get_C_first(T0s , Ts, s), new_beta)

    R = QBDFundamentalMatrices(AA, B, C, "R")
    ro = get_ro_2(lam_0, lam_1, new_beta, s0)

    u0, u10, u11 = get_bound_steady_state(R, A0, A1, AA, B, C0, ro)

    u1 = u10 + u11

    return u0, u10, u11, R

def geometric_pdf(lam0,lam1,n):
    p = lam1/(lam1+lam0)
    return p*((1-p)**(n))

def give_prob_u(u10,u11, R, u):
    return np.sum(np.dot(np.array([u10, u11]), matrix_power(R, u - 1)))

def get_steady_for_given_v(u0, u10, u11, R, v):

    steady = [u0, u10+u11]

    for steady_prob in range(2, v+2):
        steady.append(np.sum(np.dot(np.array([u10, u11]), matrix_power(R, steady_prob-1))))

    steady = np.array(steady)

    steady = np.append(steady, 1-np.sum(steady))

    return steady



def create_ph_matrix_for_each_case(event_list, lam_0, lam_1, mu_0, mu_1):

    size, size_arr = get_matrix_size(event_list)

    s = np.zeros((size, size))
    a = np.zeros(size)
    a[0] = 1

    for event_ind,  event in enumerate(event_list):

        if type(event) == str:

            if event == 'inter':
                for ind in range(size_arr[event_ind]):
                    sum_until_here = np.sum(size_arr[:event_ind])
                    s[sum_until_here+ind,sum_until_here+ind] = -(lam_0+lam_1)
                    if sum_until_here+ind<size-1:
                        s[sum_until_here + ind, sum_until_here + ind+1] = lam_0 + lam_1
            else:
                vals = event.split(',')
                lb = float(vals[0])
                ub = float(vals[1])
                for ind in range(size_arr[event_ind]):
                    sum_until_here = np.sum(size_arr[:event_ind])
                    s[sum_until_here + ind, sum_until_here + ind] = -(lam_0+lam_1+mu_0)
                    if sum_until_here+ind<size-1:
                        s[sum_until_here + ind, sum_until_here + ind+1] = lam_0+lam_1+mu_0
        else:
            if event == 0:
                for ind in range(size_arr[event_ind]):
                    sum_until_here = np.sum(size_arr[:event_ind])
                    s[sum_until_here + ind, sum_until_here + ind] = -mu_0
                    if sum_until_here + ind < size - 1:
                        s[sum_until_here + ind, sum_until_here + ind + 1] = mu_0
            elif event < 0:
                for ind in range(size_arr[event_ind]):
                    sum_until_here = np.sum(size_arr[:event_ind])
                    s[sum_until_here + ind, sum_until_here + ind] = -(mu_0+lam_0+lam_1)
                    if sum_until_here + ind < size - 1:
                        s[sum_until_here + ind, sum_until_here + ind + 1] = mu_0+lam_0+lam_1
            else: # i.e., event > 0
                for ind in range(size_arr[event_ind]):
                    sum_until_here = np.sum(size_arr[:event_ind])
                    if ind < size_arr[event_ind]-1:
                        s[sum_until_here + ind, sum_until_here + ind] = -(mu_0 + lam_0 + lam_1)
                        if sum_until_here + ind < size - 1:
                            s[sum_until_here + ind, sum_until_here + ind + 1] = mu_0 + lam_0 + lam_1
                    else:
                        s[sum_until_here + ind, sum_until_here + ind] = -mu_0
                        if sum_until_here + ind < size - 1:
                            s[sum_until_here + ind, sum_until_here + ind + 1] = mu_0


    return a, s


def get_matrix_size(event_list):

    total_size = 0
    sizes_list = []
    for event in event_list:
        if type(event) == str:
            if event == 'inter':
                total_size +=1
                sizes_list.append(1)
            else:
                vals = event.split(',')
                ub =  float(vals[1])
                total_size += ub
                sizes_list.append(int(ub))
        else:
            if event == 0:
                total_size += 1
                sizes_list.append(1)
            elif event < 0:
                total_size += int(-event)
                sizes_list.append(int(-event))
            else:
                total_size += event + 1
                sizes_list.append(int(event + 1))

    return int(total_size), np.array(sizes_list)

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




def give_pd(v, c):
    main_path = r'C:\Users\elira\PycharmProjects\redirected_git\redirected_call\pkl'
    full_path = os.path.join(main_path, str(v) + '_' + str(c) + '.pkl')
    df = pkl.load(open(full_path, 'rb'))
    return df


def marg_prob_c(v, c, mu_0, lam_0, lam_1, mu_1):
    u0, u10, u11, R = get_steady(lam_0, lam_1, mu_0, mu_1)
    steady_arr = get_steady_for_given_v(u0, u10, u11, R, v)

    prob = get_prob_c(c, steady_arr, v, mu_1, lam_0 + lam_1)

    return prob

def marg_prob_c_corr(v,c,t_prob):
    if c == 0:
        if t_prob.shape[0] >= v+1:
            return np.sum(t_prob[v+1:])
        else:
            return 0
    else:
        if v+1-c<t_prob.shape[0]:
            return t_prob[v+1-c]
        else:
            return 0


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

def prob_v(lam0, lam1, v):
    return (lam1 / (lam0 + lam1)) * (lam0 / (lam0 + lam1)) ** (v)


def prob_arrival(l1, l2, lam0, lam1, mu0, number):
    return number * ((mu0 / (mu0 + lam0 + lam1)) ** l1) * (((lam0 + lam1) / (mu0 + lam0 + lam1)) ** l2)

def give_cdf_point(result, mu0,mu1,lam0,lam1, xx):
    start_time = time.time()
    result['curr_cdf'] = result.apply(lambda x: give_cdf_from_df(x.mu0, x.lam0lam1, x.lam0lam1mu0, mu0,mu1,lam0,lam1, xx), axis=1)
    result['cdf_marg'] = result['prob']*result['curr_cdf']
    end_time = time.time()
    # print("--- %s seconds for one cdf---" % (end_time - start_time))
    return end_time - start_time


def give_cdf_from_df(num_mu0, numlam0lam1, nummu0lam0lam1, mu_0,mu_1,lam_0,lam_1, x):
    num_mu0 = int(num_mu0)
    numlam0lam1 = int(numlam0lam1)
    nummu0lam0lam1 = int(nummu0lam0lam1)

    size = num_mu0 + numlam0lam1 + nummu0lam0lam1 + 1

    limit1 = num_mu0
    limit2 = num_mu0 + numlam0lam1
    limit3 = size - 1

    s = np.zeros((size, size))
    for ind in range(s.shape[0]):
        if ind < limit1:
            s[ind, ind] = -mu_0
            s[ind, ind + 1] = mu_0
        elif ind < limit2:
            s[ind, ind] = -(lam_0 + lam_1)
            s[ind, ind + 1] = lam_0 + lam_1
        elif ind < limit3:
            s[ind, ind] = -(lam_0 + lam_1 + mu_0)
            s[ind, ind + 1] = lam_0 + lam_1 + mu_0
        else:
            s[ind, ind] = -mu_1

    alpha = np.zeros(size)
    alpha[0] = 1

    return 1 - np.sum(np.dot(alpha, expm(s * x)))

def get_prob_c(c, steady_state, v, mu1, lam):
    '''

    :param c: number of customers to arrive in the future
    :param steady_state: M/G/1 steady-state
    :param v: num of type 0 customers
    :param mu1: type 1 arrival rate
    :param lam: conditioned arrival rate
    :return: a vector the probabilites of b for all states under (v,c)
    '''
    b = v + 1 - c
    prob_b = 0

    if b == 0:
        prob_b += np.sum(steady_state[:2]) * prob_arrivals_during_exp(mu1, lam, 0)

    elif b == v + 1:
        prob_b += np.sum(steady_state[:2]) * tail_arrivals_during_exp(mu1, lam, v + 1)
        u_arr = np.arange(2, b + 1)
        k_arr = b - (u_arr - 1)
        prob_b += np.sum(steady_state[u_arr] * tail_arrivals_during_exp(mu1, lam, k_arr))
        prob_b += steady_state[-1]
    else:
        prob_b += np.sum(steady_state[:2]) * prob_arrivals_during_exp(mu1, lam, b)
        u_arr = np.arange(2, b + 2)
        k_arr = b - (u_arr - 1)
        prob_b += np.sum(steady_state[u_arr] * prob_arrivals_during_exp(mu1, lam, k_arr))

    return prob_b


# def get_C(T01, T02, alpha1, alpha2, s):
#     krom_sum = kroneker_sum(np.dot(T01, alpha1), np.dot(T02, alpha2))
#     krom_sum = kroneker_sum(krom_sum, np.zeros(s.shape))
#     return krom_sum
#
#
# def get_C_0(T01, T02, alpha1, alpha2, Beta):
#     krom_sum = kroneker_sum(np.dot(T01, alpha1), np.dot(T02, alpha2))
#     krom_sum = np.kron(krom_sum, Beta)
#
#     return krom_sum
#
#
# def get_B(T1, T2, s):
#     krom_sum = kroneker_sum(T1, T2)
#
#     return kroneker_sum(krom_sum, s)


# def get_A(T1, T2, Beta, s0):
#     kron_sum = kroneker_sum(np.zeros(T1.shape), np.zeros(T2.shape))
#     kron_sum = kroneker_sum(kron_sum, np.dot(s0, Beta))
#     return kron_sum


# def get_A0(T1, T2):
#     krom_sum = kroneker_sum(T1, T2)
#     return krom_sum


def kroneker_sum(G, H):
    size_g = G.shape[0]
    size_h = H.shape[0]
    return np.kron(G, np.identity(size_h)) + np.kron(np.identity(size_g), H)


def get_ro(lam1, lam_ext, mu11):
    return (lam1 + lam_ext) / mu11