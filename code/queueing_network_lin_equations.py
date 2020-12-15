import sys
import argparse
import numpy as np
import ast

from sympy.utilities.iterables import multiset_permutations
import pandas as pd
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import time
import pickle as pkl
import os



def main(args):
    n_max = args.n_max
    max_ones = args.max_ones

    mu = args.mu
    r = args.r
    number_of_classes = args.number_of_classes
    number_of_centers = args.number_of_centers
    P = np.zeros((number_of_centers, number_of_centers + 1, number_of_classes, number_of_classes))
    for ind in range(number_of_classes):
        P[ind, -1, ind, ind] = 1
        P[ind, np.absolute(1 - ind), np.absolute(1 - ind), np.absolute(1 - ind)] = 1

    pkl_name = 'pkl\df_results_nmax_9_mu_validate_same_rates_namx9.pkl'
    if os.path.exists(pkl_name):
        print('Continue an existing Dataframe')
        with open(pkl_name, 'rb') as f:
            df_results_summary = pkl.load(f)
    else:
        df_results_summary = pd.DataFrame([], columns=['mu_0_0', 'mu_0_1', 'mu_1_0', 'mu_1_1', 'r_0_0', 'r_0_1', 'n_max',
                                                   'max_mismatch', 'avg_lin_sys',
                                                   'avg_lin_server_0', 'avg_lin_server_1',
                                                   'avg_app_sys', 'avg_app_server_0', 'avg_app_server_1'])

    r0_vals = np.arange(0.9, 0.93, 0.02)
    for r0 in r0_vals:
        args.r[0, 0] = r0
        args.r[1, 1] = r0
        args.r[0, 1] = 1-r0
        args.r[1, 0] = 1-r0


        mu_vals = np.arange(5,5.1,0.5)
        for mu_ in tqdm(mu_vals):
            mu[0, 0] = mu_
            mu[1, 1] = mu_

            # check if the this run exists:
            if run_exists(df_results_summary, r, mu):
                print('Skip this run')
            else:


                df_results_summary = compute_system_avg_len(n_max, max_ones, mu, r, P, number_of_classes, number_of_centers, df_results_summary)


                with open(pkl_name, 'wb') as f:
                    pkl.dump(df_results_summary, f)

def run_exists(df, r, mu):

    if df.shape[0] == 0:
        return False
    shape = df.loc[(df['mu_0_0']==mu[0,0]) & (df['mu_0_1']==mu[0,1]) & (df['mu_1_0']==mu[1,0])&(df['mu_1_1']==mu[1,1]) & (df['r_0_0']==r[0,0])& (df['r_0_1']==r[0,1]),:].shape[0]
    if shape > 0:
        return True
    else:
        return False

def compute_system_avg_len(n_max, max_ones, mu, r, P, number_of_classes, number_of_centers, df_results_summary):


    df, all_states_0, all_states_1,combined_list,sys_size_dict_0, sys_size_dict_1 = get_states_structure(n_max, max_ones)

    lin_eq = np.zeros((df.shape[0], df.shape[0]))
    for state_ind in tqdm(range(df.shape[0])):
    ## Get A and B, outside rate
        lin_eq[state_ind, state_ind] += get_total_service_rate_out_of_state(df, state_ind, mu)
        lin_eq[state_ind, state_ind] += get_total_external_arrival(df, state_ind, mu, r, n_max, max_ones)
        # get internal arrival rates
        for server_ind in range(2):
            if is_next_exists_valid(df, state_ind, server_ind, server_ind, n_max, max_ones):
                next_state = get_next_states(df, state_ind, server_ind, server_ind)
                state_0, state_1 = give_new_states(df, next_state, state_ind, server_ind)
                new_index = get_new_index(df, state_0, state_1)
                rate_value = mu[server_ind, server_ind]
                lin_eq[state_ind, new_index] = -rate_value

        # get internal transissions rates
        for server_ind in range(number_of_classes):
            if 'empty' not in df.iloc[state_ind, server_ind]:
                curr_state = df.iloc[state_ind, server_ind]
                curr_state = ast.literal_eval(curr_state)
                if len(curr_state) == 1:
                    next_state = 'empty_' + str(server_ind)
                else:
                    next_state = curr_state[:-1]
                value_external_arrival = r[server_ind, curr_state[-1]]
                state_0, state_1 = give_new_states(df, next_state, state_ind, server_ind)
                new_index = get_new_index(df, state_0, state_1)
                lin_eq[state_ind, new_index] = -value_external_arrival

        for dest_server_ind in range(2):
            org_server_ind = np.absolute(1 - dest_server_ind)
            if 'empty' not in df.iloc[state_ind, dest_server_ind]:
                curr_state_arrive = df.iloc[state_ind, dest_server_ind]
                curr_state_arrive = ast.literal_eval(curr_state_arrive)
                if len(curr_state_arrive) == 1:
                    next_state_dest = 'empty_' + str(dest_server_ind)
                else:
                    next_state_dest = curr_state_arrive[:-1]
                class_tranfer = curr_state_arrive[-1]
                if 'empty' not in df.iloc[state_ind, org_server_ind]:
                    next_state_depart = ast.literal_eval(df.iloc[state_ind, org_server_ind])
                else:
                    next_state_depart = df.iloc[state_ind, org_server_ind]
                if is_next_exists_valid(df, state_ind, org_server_ind, class_tranfer, n_max, max_ones):  #
                    if 'empty' not in next_state_depart:
                        next_state_depart.insert(0, class_tranfer)
                    else:
                        next_state_depart = [class_tranfer]

                    service_rate = mu[org_server_ind, class_tranfer]

                    Prob_transission = P[org_server_ind, dest_server_ind, class_tranfer, class_tranfer]

                    service_rate = -service_rate * Prob_transission

                    if org_server_ind == 0:
                        state_0 = next_state_depart
                        state_1 = next_state_dest
                    else:
                        state_1 = next_state_depart
                        state_0 = next_state_dest
                    new_index = get_new_index(df, str(state_0), str(state_1))
                    lin_eq[state_ind, new_index] = service_rate

    lin_eq[0, :] = np.ones(df.shape[0])
    B = np.zeros(df.shape[0])
    B[0] = 1

    start_time = time.time()
    x = scipy.sparse.linalg.spsolve(lin_eq, B)
    # x = np.linalg.solve(lin_eq, B)
    print("--- %s seconds sparse linear solution ---" % (time.time() - start_time))

    print('Average system length in the system: ', np.sum(x * np.array(df['sys_size'])))
    print('Average system length in the server 0: ', np.sum(x * np.array(df['sys_size_server_0'])))
    print('Average system length in the server 1: ', np.sum(x * np.array(df['sys_size_server_1'])))

    df_ind = df_results_summary.shape[0]

    df_results_summary.loc[df_ind, 'mu_0_0'] = mu[0,0]
    df_results_summary.loc[df_ind, 'mu_0_1'] = mu[0, 1]
    df_results_summary.loc[df_ind, 'mu_1_0'] = mu[1, 0]
    df_results_summary.loc[df_ind, 'mu_1_1'] = mu[1, 1]
    df_results_summary.loc[df_ind, 'r_0_0'] = r[0,0]
    df_results_summary.loc[df_ind, 'r_0_1'] = r[0, 1]
    df_results_summary.loc[df_ind, 'n_max'] = n_max
    df_results_summary.loc[df_ind, 'max_mismatch'] = max_ones
    df_results_summary.loc[df_ind, 'avg_lin_sys'] = np.sum(x * np.array(df['sys_size']))
    df_results_summary.loc[df_ind, 'avg_lin_server_0'] = np.sum(x * np.array(df['sys_size_server_0']))
    df_results_summary.loc[df_ind, 'avg_lin_server_1'] = np.sum(x * np.array(df['sys_size_server_1']))

    p0 = (r[0,0]+r[1,0])/(r[0,0]+r[1,0]+r[0,1])
    mu_mg1 = np.array([mu[0,0],mu[0,1]])
    lamb = r[0,0]+r[1,0]+r[0,1]
    df_results_summary.loc[df_ind, 'avg_mg1_server_0'] = avg_sys(p0, mu_mg1, lamb)
    p0 = (r[1, 1] + r[0, 1]) / (r[1, 1] + r[0, 1] + r[1, 0])
    mu_mg1 = np.array([mu[1, 1], mu[1, 0]])
    lamb = r[1, 1] + r[0, 1] + r[1, 0]
    df_results_summary.loc[df_ind, 'avg_mg1_server_1'] = avg_sys(p0, mu_mg1, lamb)
    df_results_summary.loc[df_ind, 'avg_mg1_sys'] = df_results_summary.loc[df_ind, 'avg_mg1_server_0'] + df_results_summary.loc[df_ind, 'avg_mg1_server_1']


    if args.version_a_approx:
        mu_avg_0 = ((r[0,0] + r[1,0])/(r[0,0] + r[1,0] + r[0,1]))*mu[0,0]+((r[0,1])/(r[0,0] + r[1,0] + r[0,1]))*mu[0,1]
        mu_avg_1 = ((r[1, 1] + r[0, 1]) / (r[1, 1] + r[0, 1] + r[1, 0])) * mu[1, 1] + (
                    (r[1, 0]) / (r[1, 1] + r[0, 1] + r[1, 0])) * mu[1, 0]
    else:

        mu_avg_0 = (r[0, 0] + r[0, 1] + r[1, 0]) * (mu[0, 0] * mu[0, 1]) / (
                    r[0, 1] * mu[0, 0] + r[0, 0] * mu[0, 1] + r[1, 0] * mu[0, 1])
        mu_avg_1 = (r[1, 1] + r[1, 0] + r[0, 1]) * (mu[1, 1] * mu[1, 0]) / (
                    r[1, 0] * mu[1, 0] + r[1, 1] * mu[1, 0] + r[0, 1] * mu[1, 0])


    rho_0 = (r[0,0] + r[1,0] + r[0,1])/mu_avg_0
    rho_1 = (r[1, 1] + r[0, 1] + r[1, 0])/mu_avg_1


    mean_estimated_length_0 = get_mm1_mean_queue_lenght(rho_0, n_max)
    print('Estimated queue length in server 0: ', mean_estimated_length_0)
    df_results_summary.loc[df_ind, 'avg_app_server_0'] = mean_estimated_length_0
    mean_estimated_length_1 = get_mm1_mean_queue_lenght(rho_1, n_max)
    print('Estimated queue length in server 1: ', mean_estimated_length_1)
    df_results_summary.loc[df_ind, 'avg_app_server_1'] = mean_estimated_length_1
    df_results_summary.loc[df_ind, 'avg_app_sys'] = mean_estimated_length_0 + mean_estimated_length_1

    return df_results_summary


def avg_sys(p0, mu, lamb=1.0):
    rho = (p0 / mu[0] + (1 - p0) / mu[1]) * lamb
    expected_square = (2 * p0) / mu[0] ** 2 + (2 * (1 - p0)) / mu[1] ** 2
    expected = p0 / mu[0] + (1 - p0) / mu[1]
    avg_waiting = expected + lamb * expected_square / (2 * (1 - rho))
    avg_sys = avg_waiting * lamb
    print(lamb, rho, mu, p0)
    return avg_sys

def get_mm1_mean_queue_lenght(rho, n_max):
    return (rho + (rho ** (n_max + 1)) * (-1 - n_max * (1 - rho))) / (1 - rho)

def give_new_states(df, next_state, state_ind, server_ind):
    if server_ind == 0:
            state_0 = str(next_state)
            state_1 = df.iloc[state_ind, 1]
    else:
        state_1 = str(next_state)
        state_0 = df.iloc[state_ind, 0]
    return state_0, state_1

def get_states_structure(n_max, max_ones):
    all_states_0 = ['empty_0']
    all_states_1 = ['empty_1']
    all_states_0_str = ['empty_0']
    all_states_1_str = ['empty_1']
    sys_size = [0]
    sys_size_dict_0 = {'empty_0': 0}
    sys_size_dict_1 = {'empty_1': 0}


    for queue_length in tqdm(range(1, n_max + 1)):
        for num_ones in range(min(queue_length, max_ones) + 1):
            _, states_list_0, states_list_1 = get_all_states(np.array([queue_length - num_ones, num_ones]))
            for curr_state in states_list_0:
                all_states_0.append(curr_state)
                all_states_0_str.append(str(curr_state))
                sys_size.append(queue_length)
                sys_size_dict_0[str(curr_state)] = queue_length
            for curr_state in states_list_1:
                all_states_1.append(curr_state)
                all_states_1_str.append(str(curr_state))
                sys_size_dict_1[str(curr_state)] = queue_length

    combined_list = list(itertools.product(all_states_0_str, all_states_1_str))
    df_states_two_stations = pd.DataFrame(list(zip(all_states_0_str)), columns=['state'])
    df_states = pd.DataFrame(list(zip(all_states_0_str, sys_size)), columns=['state', 'sys_size'])
    lin_eq_steady = np.zeros((df_states.shape[0], df_states.shape[0]))

    df = pd.DataFrame(combined_list)
    df.columns = ['server_0', 'server_1']
    for ind in range(df.shape[0]):
        df.loc[ind, 'sys_size'] = sys_size_dict_0[df.loc[ind, 'server_0']] + sys_size_dict_1[df.loc[ind, 'server_1']]
        df.loc[ind, 'sys_size_by_server'] = str(sys_size_dict_0[df.loc[ind, 'server_0']]) + '_' + str(
            sys_size_dict_1[df.loc[ind, 'server_1']])
        df.loc[ind, 'sys_size_server_0'] = sys_size_dict_0[df.loc[ind, 'server_0']]
        df.loc[ind, 'sys_size_server_1'] = sys_size_dict_1[df.loc[ind, 'server_1']]

    print('%%%%%%%%%%%%%%%%%%%%%%%')
    print('Total number of states:', len(all_states_0) ** 2)
    print('%%%%%%%%%%%%%%%%%%%%%%%')

    return df, all_states_0, all_states_1, combined_list, sys_size_dict_0, sys_size_dict_1

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

def get_total_external_arrival(df, ind, mu, r, n_max, max_mis_matched):
    rate = 0
    for server in range(2):
        for class_ in range(2):
            rate += r[server, class_] * can_add_cust(df, ind, server, class_, n_max, max_mis_matched)

    return rate


def can_add_cust(df, state_ind, server, class_, n_max, max_mis_matched):
    is_matched = server == class_
    if not is_matched:
        return (int(df.loc[state_ind, 'sys_size_by_server'].split('_')[server]) < n_max) & (
                    df.loc[state_ind, 'server_' + str(server)].count(str(class_)) < max_mis_matched)
    else:

        return int(df.loc[state_ind, 'sys_size_by_server'].split('_')[server]) < n_max


def get_class_in_service(df, ind, station):
    return ast.literal_eval(df.loc[ind, 'server_' + str(station)])[0]


def get_total_service_rate_out_of_state(df, ind, mu):
    mu_out = 0
    for station in range(2):
        if 'empty' not in df.loc[ind, 'server_' + str(station)]:
            curr_class = get_class_in_service(df, ind, station)
            mu_out += mu[station, curr_class]

    return mu_out


def is_next_exists_valid(df, ind, server, added_server, n_max, max_):
    is_matched = server == added_server
    if 'empty' in df.iloc[ind, server]:
        return True
    else:
        state = ast.literal_eval(df.iloc[ind, server])
    if is_matched:
        if len(state) < n_max:
            return True
        else:
            return False
    else:
        if (len(state) < n_max) & (state.count(added_server) < max_):
            return True
        else:
            return False


def get_next_states(df, ind, server, added_server):
    if df.iloc[ind, server] == 'empty_' + str(server):
        next_state = [added_server]
    else:
        next_state = ast.literal_eval(df.iloc[ind, server])
        next_state.insert(0, added_server)  # put it first in list
    return next_state


def get_new_index(df, state_0, state_1):
    return df.loc[(df['server_0'] == state_0) & (df['server_1'] == state_1), :].index[0]


def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=np.array, help='external arrivals', default=np.array([[0.9, 0.1], [0.1, 0.9]]))
    parser.add_argument('--p', type=np.array, help='transision matrix', default=np.array([]))
    parser.add_argument('--number_of_centers', type=int, help='number of centers', default=2)
    parser.add_argument('--number_of_classes', type=int, help='number of classes', default=2)
    parser.add_argument('--mu', type=np.array, help='service rates', default=np.array([[3., 5.], [5., 3. ]]))
    parser.add_argument('--n_max', type=int, help='numerical_limit for steady-state', default=9)
    parser.add_argument('--max_ones', type=int, help='max num of customers from the wrong class', default=2)
    parser.add_argument('--df_pkl_name', type=str, help='transision matrix', default='df_summary_result_lin_total_spread_15_10_a.pkl')
    parser.add_argument('--version_a_approx', type=bool, help='which version of approximation', default=True)

    args = parser.parse_args(argv)

    return args


if __name__ =='__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
