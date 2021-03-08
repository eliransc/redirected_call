import pickle as pkl
import numpy as np


def main():
    num_options_1 = np.array([1, 2, 2])
    options_list = [num_options_1]
    upper_bound = 15
    for num_options_ind in range(upper_bound):
        curr_array = np.array([1])
        for ind in range(1, options_list[num_options_ind].shape[0]):
            curr_array = np.append(curr_array, curr_array[ind - 1] + options_list[num_options_ind][ind])
        curr_array = np.append(curr_array, curr_array[-1])
        options_list.append(curr_array)


    ## for lam_0+lam_1+mu_0
    if True:
        mulam0_lam1_vals_list = []
        mulam0_lam1_vals_list.append(np.array([0, 1, 1, 1, 1]))
        mulam0_lam1_vals_list.append(np.array([0, 1, 2, 2, 2, 3, 3, 2, 2, 2, 3, 3, 2, 2]))

        for v in range(3, 11):

            curr_new_mulam0lam1 = np.array([])
            for ind, val in enumerate(options_list[v - 1]):

                if val == 1:
                    curr_new_mulam0lam1 = np.append(curr_new_mulam0lam1, 0)
                elif ind < len(options_list[v - 1]) - 1:
                    curr_assignment = np.append(curr_new_mulam0lam1[np.sum(options_list[v - 1][:ind - 1])
                                                            :np.sum(options_list[v - 1][:ind])],
                                                mulam0_lam1_vals_list[v - 2][np.sum(options_list[v - 2][:ind])
                                                                    :np.sum(options_list[v - 2][:ind + 1])])
                    curr_new_mulam0lam1 = np.append(curr_new_mulam0lam1, curr_assignment+1)
                    if ind == len(options_list[v - 1]) - 2:
                        curr_new_mulam0lam1 = np.append(curr_new_mulam0lam1, curr_assignment+1)

            mulam0_lam1_vals_list.append(curr_new_mulam0lam1)
            # print(curr_new_lam0lam1)
            print(curr_new_mulam0lam1.shape[0])
        print('here')


    ## for lam_0+lam_1
    if True:
        lam0_lam1_vals_list = []
        lam0_lam1_vals_list.append(np.array([0, 0, 1, 1, 2]))
        lam0_lam1_vals_list.append(np.array([0, 0, 0, 1, 0, 0, 1, 1, 2, 1, 1, 2, 2, 3]))

        for v in range(3, 11):

            curr_new_lam0lam1 = np.array([])
            for ind, val in enumerate(options_list[v - 1]):

                if val == 1:
                    curr_new_lam0lam1 = np.append(curr_new_lam0lam1, 0)
                elif ind < len(options_list[v - 1]) - 1:
                    curr_assignment = np.append(curr_new_lam0lam1[np.sum(options_list[v - 1][:ind - 1])
                                                            :np.sum(options_list[v - 1][:ind])],
                                                lam0_lam1_vals_list[v - 2][np.sum(options_list[v - 2][:ind])
                                                                    :np.sum(options_list[v - 2][:ind + 1])])
                    curr_new_lam0lam1 = np.append(curr_new_lam0lam1, curr_assignment)
                    if ind == len(options_list[v - 1]) - 2:
                        curr_new_lam0lam1 = np.append(curr_new_lam0lam1, curr_assignment+1)

            lam0_lam1_vals_list.append(curr_new_lam0lam1)
            # print(curr_new_lam0lam1)
            print(curr_new_lam0lam1.shape[0])
        print('here')


    ## for mu
    if True:
        mu_vals_list = []
        mu_vals_list.append(np.array([1, 1, 0, 1, 0]))
        mu_vals_list.append(np.array([2, 2, 1, 0, 2, 1, 0, 1, 0, 2, 1, 0, 1, 0]))
        for v in range(3, 16):

            curr_new_mu = np.array([])
            for ind, val in enumerate(options_list[v-1]):

                if val == 1:
                    curr_new_mu = np.append(curr_new_mu, v)
                elif ind < len(options_list[v - 1]) - 1:
                    curr_assignment = np.append(curr_new_mu[np.sum(options_list[v-1][:ind-1])
                                                                  :np.sum(options_list[v-1][:ind])],
                                                      mu_vals_list[v - 2][np.sum(options_list[v - 2][:ind ])
                                                                          :np.sum(options_list[v - 2][:ind + 1])])
                    curr_new_mu = np.append(curr_new_mu, curr_assignment)
                    if ind == len(options_list[v - 1]) - 2:
                        curr_new_mu = np.append(curr_new_mu, curr_assignment)

            mu_vals_list.append(curr_new_mu)
            # print(curr_new_mu)
            print(curr_new_mu.shape[0])
    print('here')

if __name__ == '__main__':

    main()