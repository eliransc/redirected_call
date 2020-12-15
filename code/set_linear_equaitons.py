import sys
import argparse
import numpy as np
from sympy.utilities.iterables import multiset_permutations
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import time
import pickle as pkl
import os

def main_lin_eq(args):

    if True:
        df_summary_result = pd.DataFrame([], columns=['Arrival_0', 'Arrival_1', 'mu_0', 'mu_1', 'avg_waiting_lin', 'avg_waiting_estimated'])

        mean_sys_length, mean_estimated_length = lin_eq(args)

        print(mean_sys_length, mean_estimated_length)
        ind = df_summary_result.shape[0]
        df_summary_result.loc[ind, 'Arrival_0'] = args.r[0]
        df_summary_result.loc[ind, 'Arrival_1'] = args.r[1]
        df_summary_result.loc[ind, 'mu_0'] = args.mu[0]
        df_summary_result.loc[ind, 'mu_1'] = args.mu[1]
        df_summary_result.loc[ind, 'avg_waiting_lin'] = mean_sys_length
        df_summary_result.loc[ind, 'avg_waiting_estimated'] = mean_estimated_length

    with open('df_summary_result_lin_mu_acg_principle', 'wb') as f:
        pkl.dump(df_summary_result, f)

    if False:

        df_summary_result = pd.DataFrame([], columns=['Arrival_0', 'Arrival_1', 'mu_0', 'mu_1', 'avg_waiting_lin', 'avg_waiting_estimated'])

        for sim_ind in tqdm(range(10)):

            args.r[0] = 0.9 + sim_ind / 100
            args.r[1] = 1 - args.r[0]
            print(args.r)

            mean_sys_length, mean_estimated_length = lin_eq(args)

            print(mean_sys_length, mean_estimated_length)
            ind = df_summary_result.shape[0]
            df_summary_result.loc[ind, 'Arrival_0'] = args.r[0]
            df_summary_result.loc[ind, 'Arrival_1'] = args.r[1]
            df_summary_result.loc[ind, 'mu_0'] = args.mu[0]
            df_summary_result.loc[ind, 'mu_1'] = args.mu[1]
            df_summary_result.loc[ind, 'avg_waiting_lin'] = mean_sys_length
            df_summary_result.loc[ind, 'avg_waiting_estimated'] = mean_estimated_length

        with open('df_summary_result_lin_mu_1_1.5_mu_0_4_approx_b', 'wb') as f:
            pkl.dump(df_summary_result, f)

    if False:

        if os.path.exists(args.df_pkl_name):
            with open(args.df_pkl_name, 'rb') as f:
                df_summary_result = pkl.load(f)
                print('Starting from the last iteration')
        else:
            df_summary_result = pd.DataFrame([], columns=['Arrival_0', 'Arrival_1', 'mu_0', 'mu_1' ,'avg_waiting_lin', 'avg_waiting_estimated'])
            print('Starting from the scratch')

        mu_vals = np.arange(start=1.3, stop=3, step=0.3)
        p0 = np.arange(start=0.9, stop=1.0, step=0.01)
        # x = np.zeros((mu_vals.shape[0], p0.shape[0] ))
        # y = np.zeros((mu_vals.shape[0], p0.shape[0]))
        # real_vals = np.zeros((mu_vals.shape[0], p0.shape[0]))
        # estimated_val = np.zeros((mu_vals.shape[0], p0.shape[0]))
        for ind_mu_0, mu_val_0 in tqdm(enumerate(mu_vals)):
            for ind_mu_1, mu_val_1 in enumerate(mu_vals):
                for ind_p0, p0_val in enumerate(p0):
                    args.mu[0] = mu_val_0
                    args.mu[1] = mu_val_1
                    args.r[0] = p0_val
                    args.r[1] = 1 - p0_val
                    if check_if_exist(df_summary_result,[p0_val, 1-p0_val, mu_val_0, mu_val_1]):
                        print('This set of values already exists')
                    else:
                        print('New set of values')
                        if mu_val_0 < 1.5:
                            args.n_max = 32
                        else:
                            args.n_max = 27

                        print(args.r, args.mu)

                        mean_sys_length, mean_estimated_length = lin_eq(args)

                        ind = df_summary_result.shape[0]
                        df_summary_result.loc[ind, 'Arrival_0'] = args.r[0]
                        df_summary_result.loc[ind, 'Arrival_1'] = args.r[1]
                        df_summary_result.loc[ind, 'mu_0'] = args.mu[0]
                        df_summary_result.loc[ind, 'mu_1'] = args.mu[1]
                        df_summary_result.loc[ind, 'avg_waiting_lin'] = mean_sys_length
                        df_summary_result.loc[ind, 'avg_waiting_estimated'] = mean_estimated_length

                        with open(args.df_pkl_name, 'wb') as f:
                            pkl.dump(df_summary_result, f)

                    # x[ind_mu, ind_p0] = mu_val/args.mu[0]
                    # y[ind_mu, ind_p0] = p0_val
                    # real_vals[ind_mu, ind_p0] = mean_sys_length
                    # estimated_val[ind_mu, ind_p0] = mean_estimated_length

    # ax = plt.axes(projection='3d')
    #
    # ax.plot_surface(x, y, real_vals,  edgecolor='Black',
    #                 label = 'real', alpha = 1, rstride=1, cstride=1, linewidth=0.5, cmap='winter',
    #                 antialiased=True)
    # ax.plot_surface(x, y, estimated_val,  edgecolor='Red',  cmap='autumn', label = 'estimate')
    # ax.set_title('Queue Length')
    # plt.xlabel('Mu1/Mu0')
    # plt.ylabel('P0')
    #
    # plt.show()

    # with open('xyz', 'wb') as f:
    #     pkl.dump((x, y, estimated_val, real_vals), f)

    estimated_list = []
    real_list = []
    if False:
        mu_vals = np.arange(start=2, stop=4.2, step=0.1)
        for mu_val in mu_vals:
            args.mu[1] = mu_val
            mean_sys_length, mean_estimated_length = lin_eq(args)
            real_list.append(mean_sys_length)
            estimated_list.append(mean_estimated_length)
            print(mean_sys_length, mean_estimated_length)
        x_label = 'Mu_1'
        y_label = 'Avg system length'

    if False:
        p0 = np.arange(5, 11)*0.1
        for po_val in p0:
            args.r[1] = args.r[0]/po_val - args.r[0]
            print(args.r)

            mean_sys_length, mean_estimated_length = lin_eq(args)
            real_list.append(mean_sys_length)
            estimated_list.append(mean_estimated_length)
            print(mean_sys_length, mean_estimated_length)
        x_label = 'P0'
        y_label = 'Avg system length'

        fig, ax = plt.subplots()
        ax.plot(mu_vals, real_list, '-b', label='Real')
        ax.plot(mu_vals, estimated_list, '--r', label='Estimated')
        ax.set_ylabel(x_label)
        ax.set_xlabel(y_label)
        leg = ax.legend();

def get_states_structure(args):

    all_states_0 = ['empty_0']
    all_states_1 = ['empty_1']
    all_states_0_str = ['empty_0']
    sys_size = [0]
    n_max = args.n_max
    max_ones = args.max_ones


    for queue_length in tqdm(range(1, n_max + 1)):
        for num_ones in range(min(queue_length, max_ones) + 1):
            _, states_list_0, states_list_1 = get_all_states(np.array([queue_length - num_ones, num_ones]))
            for curr_state in states_list_0:
                all_states_0.append(curr_state)
                all_states_0_str.append(str(curr_state))
                sys_size.append(queue_length)
            for curr_state in states_list_1:
                all_states_1.append(curr_state)

    df_states = pd.DataFrame(list(zip(all_states_0_str, sys_size)), columns=['state', 'sys_size'])
    lin_eq_steady = np.zeros((df_states.shape[0], df_states.shape[0]))

    print('%%%%%%%%%%%%%%%%%%%%%%%')
    print('Total number of states:', len(all_states_0))
    print('%%%%%%%%%%%%%%%%%%%%%%%')

    return df_states, lin_eq_steady, all_states_0

def check_if_exist(df_mu_total, curr_list):
    if df_mu_total.shape[0] == 0:
        return 0
    for ind in range(df_mu_total.shape[0]):
        curr_list_df = []
        curr_list_df.append(df_mu_total.loc[ind, 'Arrival_0'])
        curr_list_df.append(df_mu_total.loc[ind, 'Arrival_1'])
        curr_list_df.append(df_mu_total.loc[ind, 'mu_0'])
        curr_list_df.append(df_mu_total.loc[ind, 'mu_1'])
        if curr_list_df == curr_list:
            return 1
    return 0

def lin_eq(args):

    n_max = args.n_max
    max_ones = args.max_ones
    r = args.r
    mu = args.mu

    df_states, lin_eq_steady, all_states_0 = get_states_structure(args)

    for ind, state in tqdm(enumerate(all_states_0)):

        pos = get_position(df_states, str(state))
        if state == 'empty_0':
            lin_eq_steady[ind, pos] = np.sum(r)
            for class_ in range(2):
                position = get_position(df_states, str([class_]))
                lin_eq_steady[ind, position] = -mu[class_]
        else:
            if state.count(1) == max_ones:
                indicator_arrival_1 = 0
            else:
                indicator_arrival_1 = 1
            if len(state) == n_max:
                indicator_arrival_0 = 0
            else:
                indicator_arrival_0 = 1

            lin_eq_steady[ind, pos] = r[0] * indicator_arrival_0 + r[1] * indicator_arrival_1 + mu[state[0]]
            if len(state) < n_max:
                for class_ in range(2):
                    state_ = state.copy()
                    state_.insert(0, class_)
                    if state_.count(1) <= max_ones:
                        position = get_position(df_states, str(state_))

                        lin_eq_steady[ind, position] = -mu[class_]
            if len(state) > 0:
                position = get_position(df_states, str(state[:-1]))

                lin_eq_steady[ind, position] = -r[state[-1]]
    lin_eq_steady[-1, :] = np.ones(df_states.shape[0])
    B = np.zeros(df_states.shape[0])
    B[-1] = 1


    start_time = time.time()
    x = scipy.sparse.linalg.spsolve(lin_eq_steady, B)
    print("--- %s seconds sparse linear solution ---" % (time.time() - start_time))

    mean_sparse_length = np.sum(x* np.array(df_states['sys_size']))
    print('Mean system length sparse: ', mean_sparse_length)


    p0 = r[0]/np.sum(r)
    if args.version_a_approx:
        rho = np.sum(r)/(p0*mu[0]+((1-p0)*mu[1]))
    else:
        rho = np.sum(r)/((mu[0]*mu[1])/(mu[0]-p0*mu[0]+p0*mu[1]))

    mean_estimated_length = (rho + (rho**(n_max+1))*(-1-n_max*(1-rho)))/(1-rho)
    print('Estimated queue length: ', mean_estimated_length)

    return mean_sparse_length, mean_estimated_length


def get_all_states(num_of_classes):
    '''

    :param num_of_classes: an array that specifies the number of '0' class and '1' class
    :return: all possible states for num_of_classes
    '''
    example_arr = np.concatenate(
        ((np.ones(num_of_classes[0]) * 0).astype(int), (np.ones(num_of_classes[1]) * 1).astype(int)))
    all_states = multiset_permutations(example_arr)
    size = example_arr.shape[0]
    states_array = np.zeros(size)
    states_list_0 = []
    states_list_1 = []

    for p in all_states:
        states_array = np.vstack((states_array, p))
        p1 = np.copy(np.array(p))
        zeros_postion = np.where(np.array(p) == 0)[0]
        ones_postion = np.where(np.array(p) == 1)[0]
        p1[zeros_postion] = 1
        p1[ones_postion] = 0
        states_list_0.append(p)
        states_list_1.append(list(p1))

    return states_array[1:, :], states_list_0, states_list_1


def get_position(df, state):
    if state == '[]':
        state = 'empty_0'
    return df.loc[df['state'] == state, :].index.values[0]



def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=np.array, help='external arrivals', default=np.array([0.9, 0.1]))
    parser.add_argument('--p', type=np.array, help='transision matrix', default=np.array([]))
    parser.add_argument('--number_of_centers', type=int, help='number of centers', default=1)
    parser.add_argument('--number_of_classes', type=int, help='number of classes', default=2)
    parser.add_argument('--mu', type=np.array, help='service rates', default=np.array([1, 2]))
    parser.add_argument('--n_max', type=int, help='numerical_limit for steady-state', default=250)
    parser.add_argument('--max_ones', type=int, help='max num of customers from the wrong class', default=1)
    parser.add_argument('--df_pkl_name', type=str, help='transision matrix', default='df_summary_result_lin_total_spread_15_10_a.pkl')
    parser.add_argument('--version_a_approx', type=bool, help='which version of approximation', default=False)



    args = parser.parse_args(argv)

    return args

if __name__ =='__main__':
    args = parse_arguments(sys.argv[1:])
    main_lin_eq(args)