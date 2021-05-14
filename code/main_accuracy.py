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
from create_ph_matrix import compute_ph_matrix
from get_steady_ph import get_steady_ph_sys
from compute_waiting_time import compute_waiting_time_

def main(args):

    pkl_path = r'../pkl'
    ub_high = 16
    ub_low = 16
    ub_vals = np.linspace(ub_low, ub_high, 1).astype(int)
    lam0s = np.linspace(0.5, 0.5, 1)
    total_arr = np.zeros([ub_high-ub_low+1, lam0s.shape[0]])
    for lam0_ind, lam0 in tqdm(enumerate(lam0s)):
        lam1 = 0.5


        for ind_ub_v, ub_v in enumerate(ub_vals):

            df_name_before = 'df_' + str(ub_v) + '_before_probs.pkl'
            df_name_before = os.path.join(pkl_path,df_name_before)

            df_name_after = 'df_' + str(ub_v) +'_'+str(lam0) + '_after_probs.pkl'
            df_name_after = os.path.join(pkl_path, df_name_after)

            print('stage 1: compute general structure')
            if not os.path.exists(df_name_before):
                give_number_cases(ub_v, df_name_before)
            print('stage 2: compute marginal probs')
            if not os.path.exists(df_name_after):
                compute_df(args.mu0, args.mu1, lam0, lam1, df_name_before, df_name_after, ub_v)


            df_result = pkl.load(open(df_name_after, 'rb'))

            print('stage 3: create ph matrix')
            path_ph = os.path.join(pkl_path, 'alpha_ph' +'_'+str(ub_v)+'.pkl')
            compute_ph_matrix(df_result, args.mu0, args.mu1, lam0, lam1, path_ph, ub_v)

            print('stage 4: compute steady-state')
            get_steady_ph_sys(lam1, args.lam_ext, args.mu_11, path_ph, ub_v)


            R,x = pkl.load(open('../pkl/R_' + str(ub_v) + '.pkl', 'rb'))
            print('stage 5: compute waiting time')
            compute_waiting_time_(R, x, args.mu_11, lam1, args.lam_ext, ub_v, 5)


    #         x_vals = np.linspace(0, 2, 2)
    #
    #         curr_time = []
    #
    #         time_avg = analytical_expression(df_result, args.mu0, args.mu1, lam0, lam1, [0.5,1])
    #
    #
    #         # for xx in x_vals:
    #         #     curr_time.append(give_cdf_point(df_result, args.mu0, args.mu1, lam0, lam1, xx))
    #
    #         total_arr[ind_ub_v, lam0_ind] = time_avg
    #
    #
    #         pkl.dump(total_arr, open('total_arr', 'wb'))
    # print(total_arr)


def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--ub_v', type=int, help='v_max', default=11)
    parser.add_argument('--mu0', type=float, help='mu0', default=1.666666)
    parser.add_argument('--mu1', type=float, help='mu0', default=1)
    parser.add_argument('--lam0', type=float, help='mu0', default=0.5)
    parser.add_argument('--lam1', type=float, help='mu0', default=0.5)
    parser.add_argument('--lam_ext', type=float, help='external arrival to sub queue', default=0.5)
    parser.add_argument('--mu_11', type=float, help='service rate in sub queue', default=1.25)


    args = parser.parse_args(argv)

    return args



if __name__ =='__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)