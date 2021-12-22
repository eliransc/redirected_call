# imports
import simpy
import numpy as np
import sys
import argparse
import pandas as pd
import pickle as pkl
import time
import os
from tqdm import tqdm
from datetime import datetime
from utils import *
import random

def main(args):


    # df = pd.read_excel(r'C:\Users\elira\workspace\Research\versions_settings.xlsx', sheet_name='python')
    df = pkl.load(
        open('/gpfs/fs0/scratch/d/dkrass/eliransc/redirected_git/redirected_call/code/diff_settings.pkl', 'rb'))
    ind = random.randint(0, df.shape[0]-1)


    lam00 = df.loc[ind, 'lambda00']
    lam01 = df.loc[ind, 'lambda01']

    mu00 = df.loc[ind, 'mu00']
    mu01 = df.loc[ind, 'mu01']
    mu11 = df.loc[ind, 'mu11']
    lam11 = df.loc[ind, 'lambda11']


    print('Case number: ', args.case_num)

    df_inter_departure_station_0 = pd.DataFrame([], columns = ['departure_time', 'inter_departure_time'])
    pkl.dump(df_inter_departure_station_0, open(r'../pkl/df_inter_departure_station_0_'+str(args.case_num)+'.pkl', 'wb'))

    waiting_time_list = []
    pkl.dump(waiting_time_list, open('../pkl/waiting_time_station_1_'+str(args.case_num)+'.pkl', 'wb'))

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")


    df_summary_result = pd.DataFrame([])
    for ind in tqdm(range(args.num_iterations)):

        start_time = time.time()

        with open('../pkl/avg_waiting'+str(args.case_num), 'wb') as f:
            pkl.dump(list(np.zeros(args.size)), f)

        env = simpy.Environment()

        server = []
        for server_ind in range(args.size):
            server.append(simpy.Resource(env, capacity=1))
        # p_incorrect = (1 - args.p_correct) / (args.size - 1)
        # args.r = np.identity(args.size)
        # args.r = np.where(args.r == 1, args.p_correct, p_incorrect)
        args.r = np.zeros([args.size, args.size])
        args.mu = np.zeros([args.size, args.size])

        # match_arrival = 0.6
        # mis_arrival = 0.15




        # row, col = np.diag_indices(args.r.shape[0])
        # args.r[row, col] = match_arrival
        # args.r = np.where(args.r == match_arrival, match_arrival, mis_arrival)
        args.r[0, 0] = lam00
        args.r[0, 1] = lam01
        args.r[1, 0] = 2.0
        args.r[1, 1] = lam11



        row, col = np.diag_indices(args.mu.shape[0])
        args.mu[row, col] = args.ser_matched_rate
        args.mu = np.where(args.mu == args.ser_matched_rate, args.ser_matched_rate, args.ser_mis_matched_rate)

        args.mu[0, 0] = mu00
        args.mu[0, 1] = mu01
        args.mu[1, 0] = mu10
        args.mu[1, 1] = mu11


        probabilities = (args.r / np.sum(args.r)).flatten()
        #
        # args.mu = np.identity(args.size)
        # args.mu = np.where(args.mu == 1, args.ser_matched_rate, args.ser_mis_matched_rate)



        env.process(customer_arrivals(env, server, args.r, args.mu, args.size,
                                      probabilities, args.ser_matched_rate, args.ser_mis_matched_rate, args.case_num))
        env.run(until=(args.end_time))

        with open('../pkl/avg_waiting'+str(args.case_num), 'rb') as f:
            avg_waiting = pkl.load(f)
        print(avg_waiting)

        total_avg_system = 0
        for station_ind in range(args.size):
            df_summary_result.loc[ind, 'Arrival_'+str(station_ind)] = str(args.r[station_ind])
            df_summary_result.loc[ind, 'Service_rate' + str(station_ind)] = str(args.mu[station_ind])
            df_summary_result.loc[ind, 'avg_waiting_'+str(station_ind)] = avg_waiting[station_ind]
            df_summary_result.loc[ind, 'avg_sys_'+str(station_ind)] = avg_waiting[station_ind] *\
                                                                      (np.sum(args.r[station_ind, :]) +
                                                                       np.sum(args.r[:, station_ind])
                                                                       -args.r[station_ind, station_ind])
            total_avg_system += df_summary_result.loc[ind, 'avg_sys_'+str(station_ind)]
            if station_ind == 0:
                df_summary_result.loc[ind, 'avg_sys_mg1_' + str(station_ind)], rho = avg_sys_station_0(args.r, args.mu,
                                                                                             station_ind)
                df_summary_result.loc[ind, 'avg_sys_mm1_' + str(station_ind)] = rho / (1 - rho)

            else:
                df_summary_result.loc[ind, 'avg_sys_mg1_'+str(station_ind)], rho = avg_sys(args.r, args.mu, station_ind)
                df_summary_result.loc[ind, 'avg_sys_mm1_' + str(station_ind)] = rho/(1-rho)
            # if station_ind == 0:
            #     df_summary_result.loc[ind, 'avg_sys_gg1_' + str(station_ind)] = compute_G_G_1(args.r, args.mu)

        df_summary_result.loc[ind, 'avg_sys_total'] = total_avg_system
        print(df_summary_result)

        print("--- %s seconds the %d th iteration ---" % (time.time() - start_time, 1))

        with open('../pkl/df_summary_result_sim_different_sizes_queues_'+str(current_time)+'.pkl', 'wb') as f:
            pkl.dump(df_summary_result, f)
        print('The average number of customers in station 1 is: ', df_summary_result.loc[0,'avg_sys_1'])

        df_inter_departure_station_0 = pkl.load(open(r'../pkl/df_inter_departure_station_0_' + str(args.case_num) + '.pkl', 'rb'))
        df_inter_departure_station_0 = df_inter_departure_station_0.iloc[1:, :]
        #
        # arr = np.array(df_inter_departure_station_0.loc[1:, 'inter_departure_time'])
        # arr_two_dim = np.zeros((arr.shape[0], 2))
        # for inter in range(arr.shape[0] - 1):
        #     arr_two_dim[inter, 0] = arr[inter]
        #     arr_two_dim[inter, 1] = arr[inter + 1]
        # print('The correlation is', np.corrcoef(arr_two_dim[:, 0], arr_two_dim[:,1]) )

        print('The inter-departure variance is: ',df_inter_departure_station_0['inter_departure_time'].var())

        # waiting_time_list = pkl.load(open('../pkl/waiting_time_station_1_' + str(args.case_num) + '.pkl', 'rb'))
        # wait_arr = np.array(waiting_time_list)
        # print('The 90th precentile of waiting time in station 1 is: ', np.percentile(wait_arr, 90))

        print('The average is station 0 is: ', avg_waiting[0] / (lam00+lam01))
        print('The average is station 1 is: ', df_summary_result.loc[0, 'avg_sys_1'])

        if not os.path.exists(args.df_summ):
            df = pd.DataFrame([])
        else:
            df = pkl.load(open(args.df_summ, 'rb'))
        ind = df.shape[0]
        df.loc[ind, 'lam00'] = lam00
        df.loc[ind, 'lam01'] = lam01
        df.loc[ind, 'lam10'] = lam10
        df.loc[ind, 'lam11'] = lam11

        df.loc[ind, 'mu00'] = mu00
        df.loc[ind, 'mu01'] = mu01
        df.loc[ind, 'mu10'] = mu10
        df.loc[ind, 'mu11'] = mu11

        df.loc[ind, 'avg_cust_0'] = avg_waiting[0] / (lam00+lam01)
        df.loc[ind, 'avg_cust_1'] = df_summary_result.loc[0, 'avg_sys_1']
        df.loc[ind, 'avg_wait_0'] = avg_waiting[0]
        df.loc[ind, 'avg_wait_1'] = avg_waiting[1]
        df.loc[ind,'var_0'] = df_inter_departure_station_0['inter_departure_time'].var()

        pkl.dump(df, open(args.df_summ,'wb'))

        print(df)


def avg_sys_station_0(r ,mu,ind):
    rho = r[ind,0]/mu[ind,0]+r[ind,1]/mu[ind,1]
    lamb = r[ind, 0] + r[ind, 1]
    p0 = r[ind,0]/(r[ind,0]+r[ind,1])
    expected  = p0/mu[ind,0]+(1-p0)/mu[ind,1]
    expected_2 = p0/mu[ind,0]**2+(1-p0)/mu[ind,1]**2
    avg_waiting = expected+lamb*expected_2/(2*(1-rho))
    avg_sys = avg_waiting*lamb
    return avg_sys, rho


def avg_sys(r ,mu,ind):
    p = r[ind,:]/(np.sum(r[ind,:])+np.sum(r[:,ind])-np.sum(r[ind,ind]))
    p[ind] = np.sum(r[:,ind])/(np.sum(r[ind,:])+np.sum(r[:,ind])-np.sum(r[ind,ind]))
    expected = np.sum(p/mu[ind,:])
    lamb = np.sum(r[ind,:])+np.sum(r[:,ind])-np.sum(r[ind,ind])
    rho = expected*lamb
    expected_square = np.sum(2*p/(mu[ind,:]**2))
    avg_waiting = expected + lamb * expected_square / (2 * (1 - rho))
    avg_sys = avg_waiting * lamb
    return  avg_sys, rho


def service(env, name, server, mu, arrival_time, class_, station, size, is_matched, case_num, args):
    if np.remainder(name[station], 10000) == 0:
        print('The current time is: ', env.now)
        station_ind = 1
        with open('../pkl/avg_waiting'+str(args.case_num), 'rb') as f:
            avg_waiting = pkl.load(f)
        print('The average sys in station 1 is: ',avg_waiting[station_ind] *(np.sum(args.r[station_ind, :]) +
                                                                       np.sum(args.r[:, station_ind])
                                                                       -args.r[station_ind, station_ind]))

    if (station == 0)& False:
        pkl_name = r'..\pkl\station0.pkl'
        if not os.path.exists(pkl_name):
            df = pd.DataFrame([])
            with open(pkl_name, 'wb') as f:
                pkl.dump(df, f)

        with open(pkl_name, 'rb') as f:
            df = pkl.load(f)
        ind = df.shape[0]
        df.loc[ind, 'arrival_time'] = arrival_time
        if ind > 0:
            df.loc[ind, 'inter_arrival'] = arrival_time - df.loc[ind - 1, 'arrival_time']
        df.loc[ind, 'class'] = class_
        df.loc[ind, 'queue_0'] = len(server[station].queue)+len(server[station].users)
        df.loc[ind, 'queue_1'] = len(server[1].queue) + len(server[1].users)
        df.loc[ind, 'is_matched'] = is_matched
        with open(pkl_name, 'wb') as f:
            pkl.dump(df, f)
        if np.remainder(name[station], 10000) == 0:
            print('The current time is: ', env.now)

            with open('../pkl/avg_waiting'+str(case_num), 'rb') as f:
                avg_waiting = pkl.load(f)
            print(avg_waiting)

            df_summary_result = pd.DataFrame([])

            ind = df_summary_result.shape[0]
            total_avg_system = 0
            for station_ind in range(args.size):
                df_summary_result.loc[ind, 'Arrival_' + str(station_ind)] = str(args.r[station_ind])
                df_summary_result.loc[ind, 'avg_waiting_' + str(station_ind)] = avg_waiting[station_ind]
                df_summary_result.loc[ind, 'avg_sys_' + str(station_ind)] = avg_waiting[station_ind] * \
                                                                            (np.sum(args.r[station_ind, :]) +
                                                                             np.sum(args.r[:, station_ind])
                                                                             - args.r[station_ind, station_ind])
                total_avg_system += df_summary_result.loc[ind, 'avg_sys_' + str(station_ind)]
                df_summary_result.loc[ind, 'avg_sys_mg1_' + str(station_ind)], rho = avg_sys(args.r, args.mu, station_ind)
                df_summary_result.loc[ind, 'avg_sys_mm1_' + str(station_ind)] = rho / (1 - rho)


            df_summary_result.loc[ind, 'avg_sys_total'] = total_avg_system

            now = datetime.now()

            current_time = now.strftime("%H_%M_%S")
            with open('../pkl/df_summary_result_sim_different_sizes_queues_' + str(current_time) + '.pkl', 'wb') as f:
                pkl.dump(df_summary_result, f)

    with server[station].request() as req:
        yield req


        # service time
        mu_ = mu[station, class_]
        ser_time = np.random.exponential(1 / mu_)

        yield env.timeout(ser_time)

        with open('../pkl/avg_waiting'+str(case_num), 'rb') as f:
            avg_waiting = pkl.load(f)

        waiting_time = env.now - arrival_time
        # if station == 1:
        #     waiting_time_list = pkl.load(open('../pkl/waiting_time_station_1_'+str(case_num)+'.pkl', 'rb'))
        #     waiting_time_list.append(waiting_time)
        #     pkl.dump(waiting_time_list, open('../pkl/waiting_time_station_1_'+str(case_num)+'.pkl', 'wb'))

        avg_waiting[station] = (avg_waiting[station] * name[station] + waiting_time) / (name[station] + 1)
        with open('../pkl/avg_waiting'+str(case_num), 'wb') as f:
            pkl.dump(avg_waiting, f)
        # if customer is mismatched then she is redirected to the her designated queue
        if class_ != station:
             if station == 0: # we redirect now only from station 0 now.
                station = class_
                name[station] += 1
                arrival_time = env.now
                df_inter_departure_station_0 = pkl.load(open(r'../pkl/df_inter_departure_station_0_'+str(case_num)+'.pkl', 'rb'))
                cur_ind = df_inter_departure_station_0.shape[0]
                df_inter_departure_station_0.loc[cur_ind,'departure_time'] = arrival_time
                if cur_ind > 0:
                    df_inter_departure_station_0.loc[cur_ind, 'inter_departure_time'] = arrival_time - df_inter_departure_station_0.loc[cur_ind-1, 'departure_time']
                pkl.dump(df_inter_departure_station_0, open(r'../pkl/df_inter_departure_station_0_'+str(case_num)+'.pkl', 'wb'))
                env.process(service(env, name, server, mu, arrival_time, class_, station, size, True, case_num, args))



def customer_arrivals(env, server, r, mu, size, probabilities, ser_matched_rate, ser_mis_matched_rate, case_num):

    name = np.ones(size)*(-1)

    effective_rates = np.zeros(size)
    avg_service = np.zeros(size)
    # for ind in range(size):
    #     effective_rates[ind] = np.sum(r[ind,:]) + np.sum(r[:, ind]) - r[ind, ind]
    #     avg_service[ind] = (np.sum(r[:, ind])/effective_rates[ind])*(1/ser_matched_rate)\
    #                        +(1-(np.sum(r[:, ind]))/effective_rates[ind])*(1/ser_mis_matched_rate)
    # assert np.max(effective_rates*avg_service) < 1, 'Not a  stable system'

    elements = list(np.arange(r.size))

    while True:

        # get the external stream identity
        arrival_identity = np.random.choice(elements, 1, p=probabilities)[0]

        class_ = int(np.remainder(arrival_identity, size))
        station = int(arrival_identity / size)

        curr_lamb = np.sum(r)
        yield env.timeout(np.random.exponential(1 / curr_lamb))

        arrival_time = env.now

        # update current customer

        name[station] += 1
        is_matched = station == class_
        env.process(service(env, name, server, mu, arrival_time, class_, station, size, is_matched,  case_num, args))




def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=np.array, help='external arrivals', default=np.array([]))
    parser.add_argument('--number_of_classes', type=int, help='number of classes', default=2)
    parser.add_argument('--mu', type=np.array, help='service rates', default=np.array([]))
    parser.add_argument('--end_time', type=float, help='The end of the simulation', default=350000)
    parser.add_argument('--size', type=int, help='the number of stations in the system', default=2)
    parser.add_argument('--p_correct', type=float, help='the prob of external matched customer', default=0.5)
    parser.add_argument('--ser_matched_rate', type=float, help='service rate of matched customers', default=1.2)
    parser.add_argument('--ser_mis_matched_rate', type=float, help='service rate of mismatched customers', default=10.)
    parser.add_argument('--num_iterations', type=float, help='service rate of mismatched customers', default=1)
    parser.add_argument('--case_num', type=int, help='case number in my settings', default=random.randint(0, 100000))
    parser.add_argument('--df_summ', type=str, help='case number in my settings', default='../pkl/df_sum_res_sim_4.pkl')

    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

