import argparse
import sys
import numpy as np
import pickle as pkl
import os
import pandas as pd
from tqdm import tqdm
from num_cases_recusion import give_number_cases
from compute_df_probs_ph import compute_df
from tail_analytical import analytical_expression
from utils_ph import give_cdf_point
from create_ph_matrix import compute_ph_matrix
from get_steady_ph import get_steady_ph_sys
from compute_waiting_time import compute_waiting_time_
from prob_case import compute_bayesian_probs
import time

def main(args):

    sum_results_name = 'sum_result_new1.pkl'
    pkl_path = r'../pkl'
    sum_res_full_path = os.path.join(pkl_path,sum_results_name)
    ub_high = 17
    ub_low = 17
    ub_vals = np.linspace(ub_low, ub_high, 1).astype(int)
    lam0s = np.linspace(0.5, 0.5, 1)
    total_arr = np.zeros([ub_high-ub_low+1, lam0s.shape[0]])
    start_time = time.time()

    sum_res = pd.DataFrame([],columns=('lam0','lam1','mu0','mu1','avg_station_1','inter_depart_type_1'))
    if not os.path.exists(sum_res_full_path):
        pkl.dump(sum_res, open(sum_res_full_path, 'wb'))

    for lam0_ind, lam0 in tqdm(enumerate(lam0s)):

        lam0 = 3.0
        lam1 = 1.0


        # args.lam_ext = 1-lam1
        args.mu0 = 4.15
        args.mu1 = 12.45
        args.mu_11 = 6.0



        for ind_ub_v, ub_v in enumerate(ub_vals):
            mean_num_rates_ub_v_path = os.path.join(pkl_path, str(ub_v) + '_' + str(lam0) + '_' + str(lam1) + '_' + str(
                args.mu0) + '_' + str(args.mu1) + 'mean_nam_rate_ub_v.pkl')

            df_name_before = 'df_' + str(ub_v)+'_'+str(lam0)+'_'+str(lam1) +'_'+str(args.mu0)+'_'+str(args.mu1) + '_before_probs.pkl'
            df_name_before = os.path.join(pkl_path,df_name_before)

            df_name_after = 'df_' + str(ub_v) +'_'+str(lam0)+'_'+str(lam1)+'_'+str(args.mu0)+'_'+str(args.mu1)  + '_after_probs_.pkl'
            df_name_after = os.path.join(pkl_path, df_name_after)

            print('stage 1: compute general structure')
            if not os.path.exists(df_name_before):
                give_number_cases(ub_v, df_name_before)
            print('stage 2: compute marginal probs')
            if not os.path.exists(df_name_after):
                compute_df(args.mu0, args.mu1, lam0, lam1, df_name_before, df_name_after, ub_v, mean_num_rates_ub_v_path, args, True)

            df_result = pkl.load(open(df_name_after, 'rb'))

            if args.correlation:
                print('dwq')

                compute_bayesian_probs(lam0, lam1, args.mu0, args.mu1, df_result, args.eps)

            else:
                print('stage 3: create ph matrix')
                path_ph = os.path.join(pkl_path, 'alpha_ph' +'_'+str(ub_v)+'_'+str(lam0)+'_'+str(lam1) +'_'+str(args.mu0)+'_'+str(args.mu1) +'.pkl')
                variance = compute_ph_matrix(df_result, args.mu0, args.mu1, lam0, lam1, path_ph, ub_v, mean_num_rates_ub_v_path)
                end_time = time.time()
                print('Total time for v_max = {} is: {}' .format(ub_v, (end_time-start_time)/60))

                if not args.time_check:

                    print('stage 4: compute steady-state')
                    avg_number = get_steady_ph_sys(lam1, args.lam_ext, args.mu_11, path_ph, ub_v)

                    sum_res = pkl.load(open(sum_res_full_path,'rb'))
                    ind = sum_res.shape[0]
                    sum_res.loc[ind, 'lam0'] = lam0
                    sum_res.loc[ind, 'lam1'] = lam1
                    sum_res.loc[ind, 'mu0'] = args.mu0
                    sum_res.loc[ind, 'mu1'] = args.mu1
                    sum_res.loc[ind, 'avg_station_1'] = avg_number
                    sum_res.loc[ind, 'inter_depart_type_1'] = variance

                    pkl.dump(sum_res, open(sum_res_full_path, 'wb'))

                    print(sum_res)


                    # R,x = pkl.load(open('../pkl/R_' + str(ub_v) + '.pkl', 'rb'))
                    # print('stage 5: compute waiting time')
                    # compute_waiting_time_(R, x, args.mu_11, lam1, args.lam_ext, ub_v, 6)



def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--correlation', type=bool, help='computing_correlation', default=False)
    parser.add_argument('--ub_v', type=int, help='v_max', default=11)
    parser.add_argument('--mu0', type=float, help='mu0', default=4)
    parser.add_argument('--mu1', type=float, help='mu1', default=2)
    parser.add_argument('--lam0', type=float, help='mu0', default=0.2)
    parser.add_argument('--lam1', type=float, help='mu0', default=0.8)
    parser.add_argument('--lam_ext', type=float, help='external arrival to sub queue', default=3.0)
    parser.add_argument('--mu_11', type=float, help='service rate in sub queue', default=4)
    parser.add_argument('--eps', type=float, help='error for T and U', default=0.000000001)
    parser.add_argument('--time_check', type=bool, help='do we want only the time it takes to build S', default=False)


    args = parser.parse_args(argv)

    return args

if __name__ =='__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)