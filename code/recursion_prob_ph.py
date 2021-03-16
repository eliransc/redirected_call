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

def main():

    num_options_1 = np.array([1, 2, 2])
    options_list = [num_options_1]
    upper_bound = 5
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

    for v in range(2, upper_bound):

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

                curr_assignment = np.append(curr_new_mu[np.sum(options_list[ v -1][:ind -1])
                                                        :np.sum(options_list[ v -1][:ind])],
                                            mu_vals_list[v - 2][np.sum(options_list[v - 2][:ind ])
                                                                :np.sum(options_list[v - 2][:ind + 1])])
                curr_new_mu = np.append(curr_new_mu, curr_assignment)
                if ind == len(options_list[v - 1]) - 2:
                    curr_new_mu = np.append(curr_new_mu, curr_assignment)


                curr_assignment = np.append(curr_new_lam0lam1[np.sum(options_list[v - 1][:ind - 1])
                                                              :np.sum(options_list[v - 1][:ind])],
                                            lam0_lam1_vals_list[v - 2][np.sum(options_list[v - 2][:ind])
                                                                       :np.sum(options_list[v - 2][:ind + 1])])
                curr_new_lam0lam1 = np.append(curr_new_lam0lam1, curr_assignment)
                if ind == len(options_list[v - 1]) - 2:
                    curr_new_lam0lam1 = np.append(curr_new_lam0lam1, curr_assignment +1)



                curr_assignment = np.append(curr_new_mulam0lam1[np.sum(options_list[v - 1][:ind - 1])
                                                                :np.sum(options_list[v - 1][:ind])],
                                            mulam0_lam1_vals_list[v - 2][np.sum(options_list[v - 2][:ind])
                                                                         :np.sum(options_list[v - 2][:ind + 1])])
                curr_new_mulam0lam1 = np.append(curr_new_mulam0lam1, curr_assignment +1)

                if ind == len(options_list[v - 1]) - 2:
                    curr_new_mulam0lam1 = np.append(curr_new_mulam0lam1, curr_assignment +1)

                curr_assignment = np.append(curr_new_mu0_prob[np.sum(options_list[v - 1][:ind - 1])
                                                         :np.sum(options_list[v - 1][:ind])],
                                            mu_vals_list_prob[v - 2][np.sum(options_list[v - 2][:ind])
                                                                     :np.sum(options_list[v - 2][:ind + 1])] + 1)

                curr_new_mu0_prob = np.append(curr_new_mu0_prob, curr_assignment)
                if ind == len(options_list[v - 1]) - 2:
                    curr_new_mu0_prob = np.append(curr_new_mu0_prob, curr_assignment)

                curr_assignment = np.append(curr_new_lam0lam1_prob[np.sum(options_list[v - 1][:ind - 1])
                                                              :np.sum(options_list[v - 1][:ind])] + 1,
                                            lam0_lam1_vals_list_prob[v - 2][np.sum(options_list[v - 2][:ind])
                                                                            :np.sum(options_list[v - 2][:ind + 1])])
                curr_new_lam0lam1_prob = np.append(curr_new_lam0lam1_prob, curr_assignment)
                if ind == len(options_list[v - 1]) - 2:
                    curr_new_lam0lam1_prob = np.append(curr_new_lam0lam1_prob, curr_assignment)





        mu_vals_list.append(curr_new_mu)
        lam0_lam1_vals_list.append(curr_new_lam0lam1)
        mulam0_lam1_vals_list.append(curr_new_mulam0lam1)

        mu_vals_list_prob.append(curr_new_mu0_prob)
        lam0_lam1_vals_list_prob.append(curr_new_lam0lam1_prob)

    print(curr_new_lam0lam1_prob)

if __name__ == '__main__':
    main()