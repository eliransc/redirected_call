import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import math
from scipy.linalg import expm, sinm, cosm
from scipy.integrate import quad
from utils_ph import *

def get_curr_dens(df_name_after, mu0, mu1, lam0, lam1, h):

    df = pkl.load(open(df_name_after, 'rb'))
    total_dens = 0
    for comb_ind in range(df.shape[0]):
        num_mu0 = df.loc[comb_ind, 'mu0']
        num_lam0_1 = df.loc[comb_ind, 'lam0lam1']
        num_mu0_lam0_1 = df.loc[comb_ind, 'lam0lam1mu0']

        S = create_curr_ph_inter(num_mu0, num_lam0_1,num_mu0_lam0_1, lam0, lam1, mu0, mu1)
        # curr_dens = get_density_ph(h,  S)
        curr_dens = get_density_ph(h,S)
        curr_prob = df.loc[comb_ind, 'prob']
        total_dens += curr_dens*curr_prob

    return total_dens









def get_density_ph(h,  S):

    S0 = -np.dot(S, np.ones((S.shape[0], 1)))
    alph = np.zeros(S.shape[0])
    alph[0] = 1

    return  np.dot(np.dot(alph, expm(S * h)), S0)

def get_cdf_ph(h,  S):

    alph = np.zeros(S.shape[0])
    alph[0] = 1

    return  1-np.sum(np.dot(alph, expm(S * h)))
