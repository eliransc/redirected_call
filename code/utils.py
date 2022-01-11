import argparse
import sys
# import numpy as np
# from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import *
import sys
sys.path.append(r'G:\My Drive\butools2\Python')
sys.path.append('/home/d/dkrass/eliransc/Python')
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/butools/Python')
from butools.ph import *
from butools.map import *
from butools.queues import *
from butools.mam import *


def get_mean_queue(arr):
    return np.sum(np.arange(arr.shape[0])*arr)


def get_marginal_probs_sum(limitprob, n_max):
    limit_prob_margin = np.zeros(2*n_max+1)
    for tot_num_cust in range(2*n_max + 1):
        curr_val = 0
        if tot_num_cust <= n_max:
            for num_cust_0 in range(n_max+1):
                curr_val += limitprob[num_cust_0, tot_num_cust-num_cust_0]
        else:
            for num_cust_0 in range(tot_num_cust-n_max, n_max + 1):
                curr_val += limitprob[num_cust_0, tot_num_cust-num_cust_0]
        limit_prob_margin[tot_num_cust] = curr_val


    return limit_prob_margin


def get_limit_prob_single(args, lamb_mat, num_servers, center_ind,is_multiple_servers=False, n_max=75):

    prob_arr = np.zeros(n_max)
    for num_cust in range(n_max):
        prob_arr[num_cust] = get_limit_prob_center(args, lamb_mat, center_ind, num_cust, is_multiple_servers, num_servers)

    return prob_arr

def get_limit_prob(args, lamb_mat, num_servers, is_multiple_servers=False, n_max=75):

    prob_arr = np.zeros(2*n_max)
    for num_cust in range(2*n_max):
        curr_num_cust_arr = np.zeros(num_cust + 1)
        for num_cust_in_0 in range(num_cust + 1):
            curr_num_cust_arr[num_cust_in_0] = \
                get_limit_prob_center(args, lamb_mat, 0, num_cust_in_0, is_multiple_servers, num_servers)\
               *get_limit_prob_center(args, lamb_mat, 1, num_cust - num_cust_in_0, is_multiple_servers, num_servers)

        prob_arr[num_cust] = np.sum(curr_num_cust_arr)

    prob_arr_ = np.zeros((n_max+1, n_max+1))
    for num_cust_in_0 in range(n_max+1):
        for num_cust_in_1 in range(n_max+1):
            prob_arr_[num_cust_in_0, num_cust_in_1] = get_limit_prob_center(args, lamb_mat, 0, num_cust_in_0, is_multiple_servers, num_servers)\
               *get_limit_prob_center(args, lamb_mat, 1,num_cust_in_1, is_multiple_servers, num_servers)

    return prob_arr, prob_arr_

def get_limit_prob_center(args, lamb_mat, center_ind, num_cust, is_multiple_servers, number_servers):
    rho_j = np.sum(lamb_mat[center_ind, :]) / (args.mu[center_ind]*number_servers[center_ind])

    if is_multiple_servers:
        multi_coefficient = get_multi_cofficienet(number_servers, num_cust, center_ind)
        return (rho_j**num_cust)/multi_coefficient
    else:
        rho_j = np.sum(lamb_mat[center_ind, :])/args.mu[center_ind]
        return (rho_j**num_cust)*(1-rho_j)


def get_multi_cofficienet(number_servers, num_cust, center_ind):

    coef_arr = np.zeros(num_cust)  # insert all the relative service rates
    for num_cust_ in range(num_cust):
        coef_arr[num_cust_] = service_rate_state_dependent(number_servers, center_ind, num_cust_ + 1)
    return np.prod(coef_arr)


def service_rate_state_dependent(number_servers, center_ind, num_cust_in_center):
    if num_cust_in_center < number_servers[center_ind]:
        return num_cust_in_center
    return number_servers[center_ind]


def eval_lamnda(args):
    '''

    :param args:
    :return: all the comulative arrivals per center per class
    '''
    A = get_lin_param(args.number_of_centers, args.number_of_classes, args.p)
    B = get_lin_vals(args.r)
    lambd_matrix = np.linalg.solve(A,B).reshape((args.number_of_centers, args.number_of_classes))

    return lambd_matrix

def get_lin_vals(r):
    k = r.shape[0]
    B = np.zeros((sum(r.shape), 1))
    for row in range(r.shape[0]):
        for col in range(r.shape[1]):
            B[row * k + col, 0] = r[row, col]
    return B


def get_lin_param(k, l, p):

    lamb_mat = np.zeros((k, k))
    A = np.zeros((sum(lamb_mat.shape), sum(lamb_mat.shape)))
    for j in range(k):
        for c in range(l):
            for i in range(k):
                for c1 in range(l):
                    if j * k + c == k * i + c1:
                        A[j * k + c, k * i + c1] = 1 - p[i, j, c1, c]
                    else:
                        A[j * k + c, k * i + c1] = -  p[i, j, c1, c]
    return A


def init_p_matrix(args):
    '''

    :param args:
    :return: the transion matrix
    '''
    p = np.zeros((args.number_of_centers + 1 , args.number_of_centers +1, args.number_of_classes, args.number_of_classes))
    p[0, 2, 0, 0] = 1  # from center zero to out with prob 1 for class 0
    p[1, 2, 1, 1] = 1  # from center 1 to out with prob 1 for class 1
    p[0, 1, 1, 1] = 1  # from center 0 to center 1, with prob 1 for class 1
    p[1, 0, 0, 0] = 1  # from center 1 to center 0, with prob 1 for class 0

    return p


def busy(s,lam2,mu2):
    return ((lam2+mu2+s)-((lam2+mu2+s)**2-4*lam2*mu2)**0.5)/(2*lam2)
def ser_lap(s,mu):
    return mu/(s+mu)
def hyper(s,lam1,lam2,mu1,mu2):
    return ser_lap(s,mu1)*lam1/(lam1+lam2)+ser_lap(s,mu2)*lam2/(lam1+lam2)
def rho(lam1,lam2,mu1,mu2):
    return (lam1+lam2)*((lam1/((lam1+lam2)*mu1))+(lam2/((lam1+lam2)*mu2)))
def w_lap(s,lam1,lam2,mu1,mu2):
    return ((1-rho(lam1,lam2,mu1,mu2))*s)/(s-(lam1+lam2)*(1-hyper(s,lam1,lam2,mu1,mu2)))
def F(s,lam1,lam2,mu1,mu2):
    return w_lap(s,lam1,lam2,mu1,mu2)*ser_lap(s,mu1)
def A(s,lam1,lam2,mu2):
    return(lam1/(lam1+lam2-lam2*(ser_lap(s,mu2))))
def beta(s,lam1,lam2,mu1,mu2):
    return (lam1/(lam1+lam2+s)+((A(s,lam1,lam2,mu2)*lam2)/(lam1+lam2+s))*(ser_lap(s,mu2)-busy(s+lam1,lam2,mu2)))/(1-((lam2*busy(s+lam1,lam2,mu2))/(lam1+lam2+s)))
def tau(s,lam1,lam2,mu1,mu2):
    return ser_lap(s,mu1)*(A(s,lam1,lam2,mu2)*(1-F(lam1+lam2-lam2*busy(s+lam1,lam2,mu2),lam1,lam2,mu1,mu2))+F(lam1+lam2-lam2*busy(s+lam1,lam2,mu2),lam1,lam2,mu1,mu2)*beta(s,lam1,lam2,mu1,mu2))
def get_var(lam1, lam2, mu1, mu2):
    s = Symbol('s')
    y = tau(s,lam1,lam2,mu1,mu2)
    dx = diff(y, s)
    dxdx = diff(dx,s)
    return dxdx.subs(s, 0)-(dx.subs(s,0))**2
def get_nth_moment(lam1, lam2, mu1, mu2, n):
    s = Symbol('s')
    y = tau(s,lam1,lam2,mu1,mu2)
    for i in range(n):
        if i == 0:
            dx = diff(y,s)
        else:
            dx = diff(dx,s)
    return dx.subs(s, 0)

def get_first_n_moments(parameters, n = 5 ):
    lam1, lam2, mu1, mu2 = parameters
    moments = []
    for n in range(1,n+1):
        moments.append(get_nth_moment(lam1, lam2, mu1, mu2, n)*(-1)**n)
    moments = np.array([moments], dtype='float')
    return moments
def kroneker_sum(G,H):
    size_g = G.shape[0]
    size_h = H.shape[0]
    return np.kron(G, np.identity(size_h)) + np.kron( np.identity(size_g),H)

def give_boundry_probs(R, A0, A1,A, B, C0 , rho_):
    p00, p01,p02, p10, p11, p12 = symbols('p00 p01 p02 p10 p11 p12')
    eqns = [np.dot(np.array([p00, p01,p02]),np.ones((R.shape[0])))-(1-rho_)]
    eq3 = np.dot(np.array([p00, p01, p02]), A0)+np.dot(np.array([p10, p11, p12]), A1)
    eq1 = np.dot(np.array([p00, p01, p02]), C0)
    eq2 = np.dot(np.array([p10, p11, p12]), B+np.dot(R,A))
    for eq_ind in range(R.shape[0]):
        eqns.append(eq1[0, eq_ind]+eq2[0,eq_ind])

    for eq_ind in range(R.shape[0]):
        eqns.append(eq3[0, eq_ind])

    A_mat, b = linear_eq_to_matrix(eqns[:-1], [p00, p01, p02, p10, p11, p12])
    return A_mat, b

def get_expect_gph_system(R, p1_arr, xm_max = 5000):
    expected = 0
    for pi_val in range(1,xm_max):
        ui = p1_arr.reshape((1, R.shape[0]))
        Ri = np.linalg.matrix_power(R,pi_val-1)
        expected += np.dot(np.dot(ui,Ri),np.ones((R.shape[0],1)))*pi_val
    return   expected[0,0]

def get_expect_gph_system(R, p1_arr, xm_max = 5000):
    expected = 0
    for pi_val in range(1,xm_max):
        ui = p1_arr.reshape((1, R.shape[0]))
        Ri = np.linalg.matrix_power(R,pi_val-1)
        expected += np.dot(np.dot(ui,Ri),np.ones((R.shape[0],1)))*pi_val
    return   expected[0,0]

def compute_G_G_1(r, mu):
    # Creating T's matirces


    parameters = (r[1, 0], r[1, 1], mu[1, 0], mu[1, 1])
    moments = get_first_n_moments(parameters)


    alph2, T2 = PH3From5Moments(moments[0])
    alph1 = 1.
    s = np.array([-mu[0, 0]])
    beta = 1.  # only because it is only single class now - to be changed
    s0 = -np.dot(s, np.ones((s.shape[0], 1)))
    T1 = np.array([-r[0, 0]])
    T10 = -np.dot(np.dot(T1, np.ones((T1.shape[0], 1))), alph1)
    T20 = -np.dot(np.dot(T2, np.ones((T2.shape[0], 1))), alph2)

    A0 = kroneker_sum(T1, T2)
    A1 = np.kron(np.identity(T2.shape[0]), s0)
    A = kroneker_sum(kroneker_sum(np.zeros(T1.shape), np.zeros(T2.shape)), np.dot(s0, beta))
    B = kroneker_sum(kroneker_sum(T1, T2), s)
    C = kroneker_sum(kroneker_sum(T10, T20), np.zeros(s.shape[0]))
    C0 = np.dot(kroneker_sum(T10, T20), beta)

    R = QBDFundamentalMatrices(A, B, C, "R")
    rho_ = (np.sum(r[:, 0]) / (r[0, 1] + np.sum(r[:, 0]))) * np.sum(r[:, 0]) / mu[0, 0] + (
                r[0, 1] / (r[0, 1] + np.sum(r[:, 0]))) * np.sum(r[:, 0]) / mu[0, 1]

    A_mat, b = give_boundry_probs(R, A0, A1, A, B, C0, rho_)
    p00, p01, p02, p10, p11, p12 = np.linalg.solve(np.array(A_mat, dtype=np.float), np.array(b, dtype=np.float))

    return get_expect_gph_system(R, np.array([p10, p11, p12]))



