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
import random

def main(args):


    lam0 = args.lam0
    lam1 = args.lam1


    sum_results_name = 'sum_result_2.pkl'
    pkl_path = r'../pkl'
    sum_res_full_path = os.path.join(pkl_path,sum_results_name)
    ub_high = 8
    ub_low = 8
    ub_vals = np.linspace(ub_low, ub_high, 1).astype(int)

    if sys.platform == 'linux':

        df_ = pd.read_excel('../files/corr_settings4.xlsx', sheet_name='Sheet8')


            # if os.path.exists('/scratch/d/dkrass/eliransc/inter_departure/redirected_call/pkl/util0_res.xlsx'):
            #     df = pd.read_excel('/scratch/d/dkrass/eliransc/inter_departure/redirected_call/pkl/util0_res.xlsx', sheet_name='Sheet2')
            # elif os.path.exists('/home/eliransc/projects/def-dkrass/eliransc/inter_departure/redirected_call/pkl/util0_res.xlsx'):
            #     df = pd.read_excel('/home/eliransc/projects/def-dkrass/eliransc/inter_departure/redirected_call/pkl/util0_res.xlsx',sheet_name='Sheet2')
    else:
        # df = pd.read_excel(r'C:\Users\user\workspace\redirected_call\files\corr_settings.xlsx', sheet_name='Sheet1')
        df_ = pd.read_excel('../files/corr_settings4.xlsx', sheet_name='Sheet9')

    # df = pkl.load(open('/gpfs/fs0/scratch/d/dkrass/eliransc/redirected_git/redirected_call/pkl/diff_settings_util0.pkl', 'rb'))

    for ind in tqdm(([11])):

        lam0 = df_.loc[ind, 'lambda00']
        lam1 = df_.loc[ind, 'lambda01']

        args.mu0 = df_.loc[ind, 'mu00']
        args.mu1 = df_.loc[ind, 'mu01']

        print(lam0,lam1, args.mu0, args.mu1)


        hu_0list = [0.1,1,2,5]
        cond_dict = {}
        for h_0 in hu_0list:
            h0 = h_0
            # args.mu0 = mu_0
            t1_path = '../pkl/' + str(lam0) + '_' + str(lam1) + '_' + str(args.mu0) + '_' + str(args.mu1) + '_' + str(
                h0) + 't1_dict.pkl'

            sum_res = pd.DataFrame([],columns=('lam0','lam1','mu0','mu1','avg_station_1','inter_depart_type_1'))
            if not os.path.exists(sum_res_full_path):
                pkl.dump(sum_res, open(sum_res_full_path, 'wb'))


            for ind_ub_v, ub_v in enumerate(ub_vals):

                t_prob_path = '../pkl/' + str(ub_v) + '_' + str(lam0) + '_' + str(lam1) + '_' + str(
                    args.mu0) + '_' + str(args.mu1)+ '_' + str(h0) + 't_prob.pkl'

                mean_num_rates_ub_v_path = os.path.join(pkl_path, str(ub_v) + '_' + str(lam0) + '_' + str(lam1) + '_' + str(
                    args.mu0) + '_' + str(args.mu1)+ '_' + str(h0) + 'mean_nam_rate_ub_v.pkl')

                df_name_before = 'df_' + str(ub_v)+'_'+str(lam0)+'_'+str(lam1) +'_'+str(args.mu0)+'_'+str(args.mu1) + '_' + str(h0)+ '_before_probs.pkl'
                df_name_before = os.path.join(pkl_path, df_name_before)

                df_name_after = 'df_' + str(ub_v) +'_'+str(lam0)+'_'+str(lam1)+'_'+str(args.mu0)+'_'+str(args.mu1) + '_' + str(h0) + '_after_probs.pkl'
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
                if not os.path.exists(t1_path):
                    create_t_1_probs(df_name_after,  lam0, lam1, args.mu0, args.mu1, t_prob_path, h0, t1_path)



                df_name_after_non_eq = 'df_' + str(ub_v) + '_' + str(lam0) + '_' + str(lam1) + '_' + str(args.mu0) + '_' + str(
                    args.mu1)+ '_' + str(h0) + '_after_probs_non_eq.pkl'
                df_name_after_non_eq = os.path.join(pkl_path, df_name_after_non_eq)

                h_arr = np.linspace(0.01, 1, 20)
                h_arr = np.append(h_arr, np.array([1.5, 2, 3]))

                cond_list = []
                uncond_list = []

                df_eq = pkl.load(open(df_name_after, 'rb'))
                t1_prob = pkl.load(open(t1_path,'rb'))


                for h in tqdm(h_arr):

                    tot_curr_denst = 0

                    for key in t1_prob.keys():

                        compute_df(args.mu0, args.mu1, lam0, lam1, df_name_before, df_name_after_non_eq, ub_v, mean_num_rates_ub_v_path,
                                   args, False, np.array(t1_prob[key]))

                        curr_dens = get_curr_dens(df_name_after_non_eq, args.mu0, args.mu1, lam0, lam1, h)
                        curr_prob = df_eq.loc[int(key), 'baysian_prob']
                        tot_curr_denst += curr_dens*curr_prob

                    cond_list.append(tot_curr_denst)

                    uncod_dens = get_curr_dens(df_name_after, args.mu0, args.mu1, lam0, lam1, h)
                    uncond_list.append(uncod_dens)

                cond_dict[h_0] = cond_list


                print('the end')



                cond = []
                for val_cond in cond_list:
                    #     print(val_cond[0])
                    cond.append(val_cond[0])
                cond_arr = np.array(cond)

                uncond = []
                for val_uncond in uncond_list:
                    #     print(val_uncond[0])
                    uncond.append(val_uncond[0])
                uncond_arr = np.array(uncond)

                cond_arr_norm = cond_arr / np.sum(cond_arr)
                uncond_arr_norm = uncond_arr / np.sum(uncond_arr)

                curr_kl = np.sum(uncond_arr_norm * np.log(uncond_arr_norm / cond_arr_norm))

                if not os.path.exists(args.kl_pd_path):
                    df = pd.DataFrame([],columns = ['lam0', 'lam1', 'mu0', 'mu1', 'h', 'KL'])
                    pkl.dump(df, open(args.kl_pd_path,'wb'))
                else:
                    df = pkl.load(open(args.kl_pd_path, 'rb'))

                curr_row = df.shape[0]
                df.loc[curr_row,'lam0'] = lam0
                df.loc[curr_row, 'lam1'] = lam1
                df.loc[curr_row, 'mu0'] = args.mu0
                df.loc[curr_row, 'mu1'] = args.mu1
                df.loc[curr_row, 'h'] = h_0
                df.loc[curr_row, 'KL'] = curr_kl

                pkl.dump(df, open(args.kl_pd_path,'wb'))

        pkl.dump((h_arr, cond_dict, uncond_list), open(
            '../pkl/h_arr_cond_dist_uncond_dist' + str(lam0) + '_' + str(lam1) + '_' + str(args.mu0) + '_' + str(
                args.mu1) + '_' + str(hu_0list[0])+'_'+str(hu_0list[-1])+ '.pkl', 'wb'))

        plt.figure()
        for key in cond_dict.keys():
            plt.plot(h_arr, cond_dict[key], label='$w_e$ = ' + str(key),   alpha=0.6, linewidth=3)

        plt.plot(h_arr, uncond_list, label='Equilibrium', alpha=0.9,  linewidth=3, linestyle='dashed')
        plt.legend()
        plt.savefig(
            'cond_dist' + str(lam0) + '_' + str(lam1) + '_' + str(args.mu0) + '_' + str(args.mu1) + '_' + str(hu_0list[0])+'_'+str(hu_0list[-1]) + '.png')
        # plt.show()

        max_ind = 10
        plt.figure()
        for key in cond_dict.keys():
            plt.plot(h_arr[:max_ind], cond_dict[key][:max_ind], label='$w_e$ = ' + str(key), alpha=0.6, linewidth=3)

        plt.plot(h_arr[:max_ind], uncond_list[:max_ind], label='Equilibrium', alpha=0.9, linewidth=3, linestyle='dashed')
        plt.legend()
        plt.savefig(
            'cond_dist' + str(lam0) + '_' + str(lam1) + '_' + str(args.mu0) + '_' + str(args.mu1) + '_' + str(
                hu_0list[0]) + '_' + str(hu_0list[-1]) + 'max_ind'+str(max_ind) +  '.png')
        # plt.show()

        max_ind = 20
        plt.figure()
        for key in cond_dict.keys():
            plt.plot(h_arr[:max_ind], cond_dict[key][:max_ind], label='$w_e$ = ' + str(key), alpha=0.6, linewidth=3)

        plt.plot(h_arr[:max_ind], uncond_list[:max_ind], label='Equilibrium', alpha=0.9, linewidth=3,
                 linestyle='dashed')
        plt.legend()
        plt.savefig(
            'cond_dist' + str(lam0) + '_' + str(lam1) + '_' + str(args.mu0) + '_' + str(args.mu1) + '_' + str(
                hu_0list[0]) + '_' + str(hu_0list[-1]) + 'max_ind' + str(max_ind) + '.png')
        # plt.show()

def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--correlation', type=bool, help='computing_correlation', default=True)
    parser.add_argument('--ub_v', type=int, help='v_max', default=11)
    parser.add_argument('--mu0', type=float, help='mu0', default=0.433333333)
    parser.add_argument('--mu1', type=float, help='mu1', default=4.333333333)
    parser.add_argument('--lam0', type=float, help='mu0', default=0.25)
    parser.add_argument('--lam1', type=float, help='mu0', default=0.75)
    parser.add_argument('--lam_ext', type=float, help='external arrival to sub queue', default=0.5)
    parser.add_argument('--mu_11', type=float, help='service rate in sub queue', default=1.2)
    parser.add_argument('--eps', type=float, help='error for T and U', default=0.000001)
    parser.add_argument('--kl_pd_path', type=str, help='the path to the kl pandas table', default='kl_data_4.pkl')

    args = parser.parse_args(argv)

    return args

if __name__ =='__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)