import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import math
from scipy.linalg import expm, sinm, cosm
from scipy.integrate import quad

def get_curr_dens(df_name_after, mu0, mu1, lam0, lam1, h):

    df = pkl.load(open(df_name_after, 'rb'))
    total_dens = 0
    for comb_ind in range(df.shape[0]):

        S = create_curr_ph_inter(df, comb_ind, lam0, lam1, mu0, mu1)
        curr_dens = get_density_ph(h,  S, lam0, lam1)
        curr_prob = df.loc[comb_ind, 'prob']
        total_dens += curr_dens*curr_prob

    return total_dens[0]







def create_curr_ph_inter(df_after, ind, lam_0, lam_1, mu_0, mu_1):
    num_mu0 = df_after.loc[ind, 'mu0']
    num_lam0_1 = df_after.loc[ind, 'lam0lam1']
    num_mu0_lam0_1 = df_after.loc[ind, 'lam0lam1mu0']

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

def get_density_ph(h,  S, lam0, lam1):

    S0 = -np.dot(S, np.ones((S.shape[0], 1)))
    alph = np.zeros(S.shape[0])
    alph[0] = 1

    return  np.dot(np.dot(alph, expm(S * h)), S0)
