import argparse
import sys
import numpy as np
import pickle as pkl
import os
from tqdm import tqdm
from num_cases_recusion import give_number_cases
from compute_df_probs_ph import compute_df
from tail_analytical import analytical_expression
from utils_ph import give_cdf_point


def main(args):
    ub_high = 5
    ub_low = 5
    ub_vals = np.linspace(ub_low, ub_high, 1).astype(int)
    lam0s = np.linspace(0.5, 0.5,1)
    total_arr = np.zeros([ub_high-ub_low+1, lam0s.shape[0]])
    for lam0_ind, lam0 in tqdm(enumerate(lam0s)):
        lam1 = 1-lam0


        for ind_ub_v, ub_v in enumerate(ub_vals):

            df_name_before = 'df_' + str(ub_v) + '_before_probs.pkl'
            df_name_after = 'df_' + str(ub_v) +'_'+str(lam0) + '_after_probs.pkl'

            if not os.path.exists(df_name_before):
                give_number_cases(ub_v, df_name_before)

            if not os.path.exists(df_name_after):
                compute_df(args.mu0, args.mu1, lam0, lam1, df_name_before, df_name_after)


            df_result = pkl.load(open(df_name_after, 'rb'))

            x_vals = np.linspace(0, 2, 2)

            curr_time = []

            time_avg = analytical_expression(df_result, args.mu0, args.mu1, lam0, lam1, [0.5,1])


            # for xx in x_vals:
            #     curr_time.append(give_cdf_point(df_result, args.mu0, args.mu1, lam0, lam1, xx))

            total_arr[ind_ub_v, lam0_ind] = time_avg


            pkl.dump(total_arr, open('total_arr', 'wb'))
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