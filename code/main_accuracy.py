import argparse
import sys
import numpy as np
import pickle as pkl
import os
from num_cases_recusion import give_number_cases
from compute_df_probs_ph import compute_df
from utils_ph import give_cdf_point

def main(args):
    ub_high = 21
    ub_low = 10

    lam0s = np.linspace(0.1, 0.9, 9)
    total_arr = np.zeros([ub_high-ub_low+1, lam0s.shape[0]])
    for lam0_ind, lam0 in enumerate(lam0s):
        lam1 = 1-lam0

        curr_lam_0_arr = np.array([])
        for ub_v in range(ub_low, ub_high+1):

            df_name = 'df_' + str(ub_v) + '.pkl'

            if not os.path.exists(df_name):
                give_number_cases(ub_v, df_name)

                compute_df(args.mu0, args.mu1, lam0, lam1, df_name)

            df_result = pkl.load(open(df_name, 'rb'))

            x_vals = np.linspace(0, 2, 10)

            curr_time = []
            for xx in x_vals:
                curr_time.append(give_cdf_point(df_result,args.mu0, args.mu1, lam0, lam1, xx))

            total_arr[ub_v-ub_low, lam0_ind] = np.mean(np.array(curr_time))
        #     curr_lam_0_arr = np.append(curr_lam_0_arr,np.mean(np.array(curr_time)))
        #
        # total_arr[lam0_ind, :] = curr_lam_0_arr

    print(total_arr)









def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--ub_v', type=int, help='v_max', default=11)
    parser.add_argument('--mu0', type=float, help='mu0', default=0.7)
    parser.add_argument('--mu1', type=float, help='mu0', default=3.)
    parser.add_argument('--lam0', type=float, help='mu0', default=0.5)
    parser.add_argument('--lam1', type=float, help='mu0', default=0.5)


    args = parser.parse_args(argv)

    return args



if __name__ =='__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)