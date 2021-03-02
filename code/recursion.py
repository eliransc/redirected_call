import numpy as np
from math import comb
import pickle as pkl
from tqdm import tqdm

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


def possibilites_after_initial_arrivals(num_arrivals, arrivals, services, curr_comb, combnum, pkl_name_inter_depart):

    if services == 1:

        ## computing the values to add to cuu_comb
        with open(pkl_name_inter_depart, 'rb') as f:
            count, combp = pkl.load(f)

        if arrivals == 1: # if there is only one customer to arrive then it can either during or after service
            update_crr_comb = np.array([1, 0])

            size_comb = np.append(curr_comb, update_crr_comb).shape[0]
            combp = np.append(combp, np.append(curr_comb, update_crr_comb).reshape(1, size_comb), axis=0)
            count += 1
            print(np.append(curr_comb, update_crr_comb))

            update_crr_comb = np.array([0, 1])
            combp = np.append(combp, np.append(curr_comb, update_crr_comb).reshape(1, size_comb), axis=0)
            count += 1
            print(np.append(curr_comb, update_crr_comb))


        else: # all customers arrived already
            update_crr_comb = np.array([0, 0])
            size_comb = np.append(curr_comb, update_crr_comb).shape[0]
            if combp.shape[0] == 0: # If this is the first time we add to te combination matrix
                combp = np.append(curr_comb, update_crr_comb).reshape(1, size_comb) # first combination
            else:
                combp = np.append(combp, np.append(curr_comb, update_crr_comb).reshape(1, size_comb), axis=0) # adding a further combination
            count += 1
            print(np.append(curr_comb, update_crr_comb))

        with open(pkl_name_inter_depart, 'wb') as f: # dump to pkl
            pkl.dump((count, combp), f)

        combnum += 1

    else:
        # if the
        if (services != num_arrivals) & (arrivals >= services):
            lb = 0
        else:
            lb = -1
        for ind in range(arrivals, lb, -1):

            if services == num_arrivals:

                curr_comb = np.array([])
                if ind == 0:
                    in_service = ind
                    inter_arrive = 1
                else:
                    in_service = ind
                    inter_arrive = 0

            ## computing the values to add to cuu_comb
            else:

                in_service = ind
                inter_arrive = 0

            update_crr_comb = np.array([in_service, inter_arrive])

            possibilites_after_initial_arrivals(num_arrivals, arrivals - (in_service + inter_arrive), services-1,
                                                np.append(curr_comb, update_crr_comb), combnum, pkl_name_inter_depart)

            # in case we need to decide whether arrivals occurs at service or not
            # if only one customer is left to arrive, it is also possible that she will arrive after service is done
            if (num_arrivals > services) & (ind == 1) & (arrivals == services): # cond(1): cannot happen in the first service cond(2): only if one customer should arrive cond(3): only if remianing arrival equals remianing services
                in_service = 0
                inter_arrive = 1

                update_crr_comb = np.array([in_service, inter_arrive])

                possibilites_after_initial_arrivals(num_arrivals, arrivals - (in_service + inter_arrive), services - 1,
                                                    np.append(curr_comb, update_crr_comb), combnum,
                                                    pkl_name_inter_depart)

def main():

    for num_arrivals in tqdm(range(2, 6)):

        arrivals = num_arrivals
        services = num_arrivals

        pkl_name_inter_depart = '../pkl/combs'+str(num_arrivals-1)+'.pkl'

        combs = np.array([])
        count = 0
        with open(pkl_name_inter_depart, 'wb') as f:
            pkl.dump((count, combs), f)

        combnum = 0
        curr_comb = np.array([])
        possibilites_after_initial_arrivals(num_arrivals, arrivals, services, curr_comb, combnum, pkl_name_inter_depart)

        # with open(pkl_name_inter_depart, 'rb') as f:
        #     count, combp = pkl.load(f)

        # print(combp)
        # print(count)

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