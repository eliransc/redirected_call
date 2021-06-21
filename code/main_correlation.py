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
from create_t1_probs import create_t_1_probs
from get_curr_dens_ import get_curr_dens
import matplotlib.pyplot as plt

def main(args):

    t1_path_a = '../pkl/t1_dict_a.pkl'
    t1_path_b = '../pkl/t1_dict_b.pkl'
    bayes_prob_a = '../pkl/bayes_prob_a.pkl'
    bayes_prob_b = '../pkl/bayes_prob_b.pkl'

    lam0 = args.lam0
    lam1 = args.lam1


    sum_results_name = 'sum_result20.pkl'
    pkl_path = r'../pkl'
    sum_res_full_path = os.path.join(pkl_path,sum_results_name)
    ub_high = 10
    ub_low = 10
    ub_vals = np.linspace(ub_low, ub_high, 1).astype(int)

    h0 = 1


    sum_res = pd.DataFrame([],columns=('lam0','lam1','mu0','mu1','avg_station_1','inter_depart_type_1'))
    if not os.path.exists(sum_res_full_path):
        pkl.dump(sum_res, open(sum_res_full_path, 'wb'))


    for ind_ub_v, ub_v in enumerate(ub_vals):

        t_prob_path = '../pkl/' + str(ub_v) + '_' + str(lam0) + '_' + str(lam1) + '_' + str(
            args.mu0) + '_' + str(args.mu1) + 't_prob.pkl'

        mean_num_rates_ub_v_path = os.path.join(pkl_path, str(ub_v) + '_' + str(lam0) + '_' + str(lam1) + '_' + str(
            args.mu0) + '_' + str(args.mu1) + 'mean_nam_rate_ub_v.pkl')

        df_name_before = 'df_' + str(ub_v)+'_'+str(lam0)+'_'+str(lam1) +'_'+str(args.mu0)+'_'+str(args.mu1) + '_before_probs.pkl'
        df_name_before = os.path.join(pkl_path, df_name_before)

        df_name_after = 'df_' + str(ub_v) +'_'+str(lam0)+'_'+str(lam1)+'_'+str(args.mu0)+'_'+str(args.mu1)  + '_after_probs.pkl'
        df_name_after = os.path.join(pkl_path, df_name_after)

        print('stage 1: compute general structure')
        if not os.path.exists(df_name_before):
            give_number_cases(ub_v, df_name_before)
        print('stage 2: compute marginal probs')
        if not os.path.exists(df_name_after):
            compute_df(args.mu0, args.mu1, lam0, lam1, df_name_before, df_name_after, ub_v, mean_num_rates_ub_v_path, args, True)

        # df_result = pkl.load(open(df_name_after, 'rb'))
        if not os.path.exists(t_prob_path):
            t_shape = compute_bayesian_probs(lam0, lam1, args.mu0, args.mu1,  args.eps, df_name_after, t_prob_path, h0)
        if not os.path.exists(t1_path_a):
            create_t_1_probs(df_name_after, lam0, lam1, args.mu0, args.mu1, t_prob_path, t1_path_a, t1_path_b, bayes_prob_a, bayes_prob_b, 45)

        t1_prob_a = pkl.load(open(t1_path_a, 'rb'))
        bayes_prob_a_dict = pkl.load(open(bayes_prob_a, 'rb'))


        df_name_after_non_eq = 'df_' + str(ub_v) + '_' + str(lam0) + '_' + str(lam1) + '_' + str(args.mu0) + '_' + str(
            args.mu1) + '_after_probs_non_eq.pkl'
        df_name_after_non_eq = os.path.join(pkl_path, df_name_after_non_eq)

        h_arr = np.linspace(0.000001, 3, 50)

        cond_list = []
        uncond_list = []

        for h in tqdm(h_arr):
            total_cond_dens = 0
            for key in t1_prob_a.keys():

                compute_df(args.mu0, args.mu1, lam0, lam1, df_name_before, df_name_after_non_eq, ub_v, mean_num_rates_ub_v_path,
                           args, False, np.array(t1_prob_a[key]))

                curr_dens = get_curr_dens(df_name_after_non_eq, args.mu0, args.mu1, lam0, lam1, h)
                curr_prob = bayes_prob_a_dict[key]
                total_cond_dens += curr_dens*curr_prob

            t1_prob_b = pkl.load(open(t1_path_b, 'rb'))
            bayes_prob_b_dict = pkl.load(open(bayes_prob_b, 'rb'))

            for key in t1_prob_b.keys():

                compute_df(args.mu0, args.mu1, lam0, lam1, df_name_before, df_name_after_non_eq, ub_v, mean_num_rates_ub_v_path,
                           args, False, np.array(t1_prob_b[key]))

                curr_dens = get_curr_dens(df_name_after_non_eq, args.mu0, args.mu1, lam0, lam1, h)
                curr_prob = bayes_prob_b_dict[key]
                total_cond_dens += curr_dens*curr_prob

            print(total_cond_dens)
            cond_list.append(total_cond_dens)

            uncod_dens = get_curr_dens(df_name_after, args.mu0, args.mu1, lam0, lam1, h)
            uncond_list.append(uncod_dens)
            print(uncod_dens)

        plt.figure()
        plt.plot(h_arr, cond_list, label = 'conditioned')
        plt.plot(h_arr, uncond_list, label = 'prior')
        plt.legend()
        plt.show()

        print('the end')




def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--correlation', type=bool, help='computing_correlation', default=True)
    parser.add_argument('--ub_v', type=int, help='v_max', default=11)
    parser.add_argument('--mu0', type=float, help='mu0', default=2000)
    parser.add_argument('--mu1', type=float, help='mu1', default=1.5)
    parser.add_argument('--lam0', type=float, help='mu0', default=0.5)
    parser.add_argument('--lam1', type=float, help='mu0', default=0.5)
    parser.add_argument('--lam_ext', type=float, help='external arrival to sub queue', default=0.5)
    parser.add_argument('--mu_11', type=float, help='service rate in sub queue', default=1.5)
    parser.add_argument('--eps', type=float, help='error for T and U', default=0.0000000001)


    args = parser.parse_args(argv)

    return args

if __name__ =='__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)