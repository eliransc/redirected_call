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

    sum_results_name = 'sum_result_many_4.pkl'
    pkl_path = r'../pkl'
    sum_res_full_path = os.path.join(pkl_path,sum_results_name)
    ub_high = 3
    ub_low = 3
    ub_vals = np.linspace(ub_low, ub_high, 1).astype(int)
    lam0s = np.linspace(0.5, 0.5, 1)
    total_arr = np.zeros([ub_high-ub_low+1, lam0s.shape[0]])
    start_time = time.time()

    sum_res = pd.DataFrame([],columns=('lam0','lam1','mu0','mu1','avg_station_1','inter_depart_type_1'))
    if not os.path.exists(sum_res_full_path):
        pkl.dump(sum_res, open(sum_res_full_path, 'wb'))

    # df = pd.read_excel(r'C:\Users\elira\workspace\Research\sum_results_rho0.xlsx', sheet_name='python_util0')
    # # df = pkl.load(
    # #     open('/gpfs/fs0/scratch/d/dkrass/eliransc/redirected_git/redirected_call/code/diff_settings.pkl', 'rb'))

    if sys.platform == 'linux':
        df = pd.read_excel('../files/corr_settings_1.xlsx', sheet_name='Sheet2')
    else:
        df = pd.read_excel(r'C:\Users\user\workspace\redirected_call\files\corr_settings.xlsx', sheet_name='Sheet3')


    for ind in range(0,2):

        lam0 = df.loc[ind,'lambda00']
        lam1 = df.loc[ind,'lambda01']


        args.mu0 = df.loc[ind,'mu00']
        args.mu1 = df.loc[ind,'mu01']
        args.mu_11 = df.loc[ind,'mu11']
        args.lam_ext = df.loc[ind, 'lambda11']

        curr_ind = ind


        if lam0 == 0.25:
           ub_v = 9
        elif lam0 == 1:
            ub_v = 12
        else:
            ub_v = 20



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
                sum_res.loc[ind, 'lam11'] =  args.lam_ext
                sum_res.loc[ind, 'mu11'] = args.mu_11
                sum_res.loc[ind, 'avg_station_1'] = avg_number
                sum_res.loc[ind, 'inter_depart_type_1'] = variance
                rho1 = (lam1+args.lam_ext)/args.mu_11
                sum_res.loc[ind, 'Pois_avg_station_1'] = rho1/(1-rho1)
                sum_res.loc[ind, 'Pois_Var'] = 1/lam1**2
                sum_res.loc[ind, 'ind'] = curr_ind


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
    parser.add_argument('--lam_ext', type=float, help='external arrival to sub queue', default=0.333333333)
    parser.add_argument('--mu_11', type=float, help='service rate in sub queue', default=4)
    parser.add_argument('--eps', type=float, help='error for T and U', default=0.000000001)
    parser.add_argument('--time_check', type=bool, help='do we want only the time it takes to build S', default=False)


    args = parser.parse_args(argv)

    return args

if __name__ =='__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)