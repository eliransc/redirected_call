import numpy as np
import pickle as pkl
from tqdm import tqdm


def main():
    pkl_name_inter_depart = '../pkl/combs.pkl'
    total_ph_lists = []
    with open(pkl_name_inter_depart, 'rb') as f:
        count, combp = pkl.load(f)
    print(combp)
    v = 2
    for curr_ind in tqdm(range(combp.shape[0])):
        comb = combp[curr_ind,:]
        ph = []

        if comb[1] == 1:
            ph.append('inter')
        for ph_ind in range(2, comb.shape[0]):
            if ph_ind % 2 == 0:
                if np.sum(comb[ph_ind:]) == 0:
                    ph.append(0)
                else:
                    # if ph_ind < comb.shape[0]-1:
                    if (comb[ph_ind] > 0) & (np.sum(comb[ph_ind+1:]) == 0):
                        ph.append(comb[ph_ind])
                    elif (comb[ph_ind] == 0) & (np.sum(comb[ph_ind+1:]) > 0):
                        ph.append(-1)
                    else:
                        curr_str = str(comb[ph_ind])
                        curr_str = curr_str+',' + str(comb[ph_ind]+1)
                        ph.append(curr_str)

            else:
                if comb[ph_ind] == 1:
                    ph.append('inter')

        total_ph_lists.append(ph)

    for lis in total_ph_lists:

        print(lis)


if __name__ == '__main__':

    main()