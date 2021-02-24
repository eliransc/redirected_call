import numpy as np
from math import comb
import pickle as pkl

def give_possible_options(combs, num_arrivals, num_services, arrivals, services, combnum, curr_comb, pkl_name_inter_depart):


    if services == 1:

        curr_comb = np.append(curr_comb, arrivals)

        with open(pkl_name_inter_depart, 'rb') as f:
            count, combp = pkl.load(f)
        combp[count, :] = curr_comb
        count += 1

        with open(pkl_name_inter_depart, 'wb') as f:
            pkl.dump((count, combp), f)

        combs[combnum, num_services - services] = arrivals
        combnum += 1

    else:

        for ind in range(arrivals, -1, -1):

            if services == num_services:

                curr_comb = np.array([])

            give_possible_options(combs, num_arrivals, num_services, arrivals - ind,
                                                       services-1, combnum, np.append(curr_comb, ind), pkl_name_inter_depart)

    return combs


def possibilites_after_initial_arrivals(num_arrivals, arrivals, services, curr_comb, combnum):

    if services == 1:

        pass

    else:

        for ind in range(arrivals, -1, -1):

            if services == num_arrivals:

                curr_comb = np.array([])



def main():

    num_arrivals = 3
    arrivals = num_arrivals
    services = num_arrivals

    pkl_name_inter_depart = '../pkl/combs.pkl'


    combs = np.array([])
    count = 0
    with open(pkl_name_inter_depart, 'wb') as f:
        pkl.dump((count, combs), f)

    combnum = 0
    curr_comb = np.array([])
    possibilites_after_initial_arrivals(num_arrivals, arrivals, services, curr_comb, combnum)

    if False:

        num_arrivals = 3
        num_services = 4
        arrivals = num_arrivals
        services = num_services

        pkl_name_inter_depart = '../pkl/combs.pkl'

        num_possi_coms = comb(num_arrivals + num_services - 1, num_services - 1)
        combs = np.zeros((num_possi_coms, num_services))
        count = 0
        with open(pkl_name_inter_depart, 'wb') as f:
            pkl.dump((count,combs), f)


        combnum = 0
        curr_comb = np.array([])
        give_possible_options(combs, num_arrivals, num_services, arrivals, services, combnum,
                              curr_comb, pkl_name_inter_depart)

        with open(pkl_name_inter_depart, 'rb') as f:
            count, combp = pkl.load(f)
        print(combp)


if __name__ == '__main__':

    main()