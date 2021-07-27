

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import math
from scipy.linalg import expm, sinm, cosm
from scipy.integrate import quad
from utils_ph import *

def compute_bayesian_probs(lam_0,lam_1,mu_0,mu_1, eps, df_name_after, t_prob_path, h):

    df_after = pkl.load(open(df_name_after, 'rb'))

    u0, u10, u11, R = get_steady(lam_0, lam_1, mu_0, mu_1)

    u_prob = [u0, u10 + u11]
    for u in range(2, 2000):
        curr_prob = give_prob_u(u10, u11, R, u)

        u_prob.append(curr_prob)

        if curr_prob < eps:
            break
    t1_max = u
    u_prob = np.array(u_prob)
    # print(u)

    t_prob = []
    # t1_max = 2000
    s = np.array([-mu_1]).reshape((1, 1))
    for t1 in range(t1_max):
        curr_prob = 0
        for u in range(t1 + 1 + 1):
            if u == 0:
                t = t1
            else:
                t = t1 - u + 1

            numirator = u_prob[u] * quad(get_density, 0, 100, args=(s, lam_0, lam_1, t,))[0] \

            if numirator > 0:
                curr_prob += numirator/math.factorial(t)

        t_prob.append(curr_prob)

        if curr_prob < eps:
            break

    t_prob = np.array(t_prob)

    pkl.dump(t_prob, open(t_prob_path, 'wb'))


    df_after['dens_h'] = df_after.apply(lambda x: get_params_dens(x.mu0, x.lam0lam1, x.lam0lam1mu0, h, lam_0, lam_1, mu_0, mu_1), axis=1)

    df_after['dens_prob'] = df_after['dens_h']*df_after['prob']

    df_after['baysian_prob'] = df_after['dens_prob']/df_after['dens_prob'].sum()

    print('fef')

    df_after['num_mu_0'] = df_after.apply(lambda x: num_mu_0(x.v, x.c, x.Ar), axis=1)
    df_after['num_mu_1'] = df_after.apply(lambda x: num_mu_1(x.Ar), axis=1)

    pkl.dump(df_after, open(df_name_after, 'wb'))

    return t_prob.shape[0]



def get_dens_mixture(h, ph_list, df_after):
    total_dens = 0,
    for ind_ph, ph in enumerate(ph_list):
        curr_prob = df_after.loc[ind_ph, 'prob']
        curr_dens = get_density_ph(h, ph)
        total_dens += curr_prob * curr_dens

    return total_dens

def get_density(h,  S, lam0, lam1,  k):

    S0 = -np.dot(S, np.ones((S.shape[0], 1)))
    alph = np.zeros(S.shape[0])
    alph[0] = 1

    val1 = np.exp(-(lam0 + lam1) * h)
    val3 = np.dot(np.dot(alph, expm(S * h)), S0)

    if k <150:
        val2 = (((lam0 + lam1) * h) ** k)
    else:
        val2 = 0
    return val1 * val2 * val3



    # return np.exp(-(lam0 + lam1) * h) * (((lam0 + lam1) * h) ** k) * np.dot(np.dot(alph, expm(S * h)), S0)

def comp_t1(t1_prob, t1):
    return t1_prob[t1]

def get_params_dens(num_mu0, num_lam0_1, num_mu0_lam0_1, h, lam_0, lam_1, mu_0,mu_1):

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

    return get_density_ph(h, ph)[0]

def get_density_ph(h, S):
    S0 = -np.dot(S, np.ones((S.shape[0], 1)))
    alph = np.zeros(S.shape[0])
    alph[0] = 1

    return np.dot(np.dot(alph, expm(S * h)), S0)


def num_mu_0(v, c, ar):
    if v == 0:
        return 0

    else:
        if ar == 0:
            return v
        elif ar == v + 1:
            return 0
        else:
            return v - ar + 1


def num_mu_1(ar):
    if ar == 0:
        return 2
    else:
        return 1