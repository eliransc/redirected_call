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




init_path = '/home/eliransc/projects/def-dkrass/eliransc/redirected_call/code/init_list_37.pkl'
if not os.path.exists(init_path):
    pkl.dump(np.arange(30), open(init_path, 'wb'))

initial_list = pkl.load(open(init_path, 'rb'))
case_ind = np.random.choice(initial_list)
initial_list = np.delete(initial_list, np.where(initial_list == case_ind))
pkl.dump(initial_list, open(init_path, 'wb'))

def main(args):

    if sys.platform == 'linux':

        df = pd.read_excel('../files/util0_res.xlsx', sheet_name='Sheet12')
    else:
        df = pd.read_excel('../files/corr_settings4.xlsx', sheet_name='Sheet8')



    print(case_ind)

    lam00 = df.loc[case_ind, 'lambda00']
    lam01 = df.loc[case_ind, 'lambda01']

    mu00 = df.loc[case_ind, 'mu00']
    mu01 = df.loc[case_ind, 'mu01']
    mu11 = df.loc[case_ind, 'mu11']
    lam11 = df.loc[case_ind, 'lambda11']

    mu10 = 2.0
    lam10 = 0.0

    print('Case number: ', args.case_num)

    df_inter_departure_station_0 = pd.DataFrame([], columns = ['departure_time', 'inter_departure_time'])
    inter_dep_path = r'../pkl/df_inter_departure_station_0_case_ind_' + str(case_ind) + '_' + str(
        args.case_num) + '.pkl'
    pkl.dump(df_inter_departure_station_0, open(inter_dep_path, 'wb'))

    waiting_time_list = []
    pkl.dump(waiting_time_list, open('../pkl/waiting_time_station_1_'+str(args.case_num)+'.pkl', 'wb'))

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")

    df_summary_result = pd.DataFrame([])
    for ind in tqdm(range(args.num_iterations)):

        start_time = time.time()

        env = simpy.Environment()

        server = []
        for server_ind in range(args.size):
            server.append(simpy.Resource(env, capacity=1))

        args.r = np.zeros([args.size, args.size])
        args.mu = np.zeros([args.size, args.size])

        args.r[0, 0] = lam00
        args.r[0, 1] = lam01
        args.r[1, 0] = 0.00
        args.r[1, 1] = lam11



        row, col = np.diag_indices(args.mu.shape[0])
        args.mu[row, col] = args.ser_matched_rate
        args.mu = np.where(args.mu == args.ser_matched_rate, args.ser_matched_rate, args.ser_mis_matched_rate)

        args.mu[0, 0] = mu00
        args.mu[0, 1] = mu01
        args.mu[1, 0] = mu10
        args.mu[1, 1] = mu11


        probabilities = (args.r / np.sum(args.r)).flatten()


        sums = [0,0,0,0,0,0,0,0]

        corr_time = [0]
        corr_path = '../pkl/corr_time'+str(args.case_num)+'.pkl'
        pkl.dump(corr_time, open(corr_path, 'wb'))

        avg_time = list(np.zeros(args.size))

        waiting_path = '../pkl/waiting_time' + str(args.case_num) + '.pkl'

        pkl.dump([0,0], open(waiting_path, 'wb'))

        env.process(customer_arrivals(env, server, args.r, args.mu, args.size,
                                      probabilities, args.ser_matched_rate, args.ser_mis_matched_rate, args.case_num, sums, avg_time))
        env.run(until=(args.end_time))

        avg_waiting = pkl.load(open(waiting_path, 'rb'))

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


        df_summary_result.loc[ind, 'avg_sys_total'] = total_avg_system
        print(df_summary_result)

        print("--- %s seconds the %d th iteration ---" % (time.time() - start_time, 1))

        with open('../pkl/df_summary_result_sim_different_sizes_queues_'+str(current_time)+'.pkl', 'wb') as f:
            pkl.dump(df_summary_result, f)
        print('The average number of customers in station 1 is: ', df_summary_result.loc[0,'avg_sys_1'])

        print('The average is station 0 is: ', avg_waiting[0] * (lam00+lam01))
        print('The average is station 1 is: ', df_summary_result.loc[0, 'avg_sys_1'])


        if not os.path.exists(args.df_summ):
            df = pd.DataFrame([])
        else:
            df = pkl.load(open(args.df_summ, 'rb'))
        ind = df.shape[0]

        df.loc[ind, 'lam00'] = lam00
        df.loc[ind, 'lam01'] = lam01
        df.loc[ind, 'lam10'] = lam10
        df.loc[ind, 'lam11'] = 0#lam11

        df.loc[ind, 'mu00'] = mu00
        df.loc[ind, 'mu01'] = mu01
        df.loc[ind, 'mu10'] =  0 #mu10
        df.loc[ind, 'mu11'] = mu11

        df.loc[ind, 'avg_cust_0'] = avg_waiting[0] * (lam00+lam01)
        df.loc[ind, 'avg_cust_1'] = df_summary_result.loc[0, 'avg_sys_1']
        df.loc[ind, 'avg_wait_0'] = avg_waiting[0]
        df.loc[ind, 'avg_wait_1'] = avg_waiting[1]
        # df.loc[ind,'var_0'] = df_inter_departure_station_0['inter_departure_time'].var()

        df.loc[ind, 'ind'] = case_ind
        corr_time = pkl.load(open(corr_path, 'rb'))
        df.loc[ind, 'inter_rho'] = corr_time[-1]

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


def service(env, name, server, mu, arrival_time, class_, station, size, is_matched, case_num, args, sums, avg_time):
    if (np.remainder(name[station], 10000) == 0) & (station == 0):
        print('The current time is: ', env.now)
        station_ind = 1
        # with open('../pkl/avg_waiting'+str(args.case_num), 'rb') as f:
        #     avg_waiting = pkl.load(f)
        print('The average sys in station 1 is: ',avg_time[station_ind] *(np.sum(args.r[station_ind, :]) +
                                                                       np.sum(args.r[:, station_ind])
                                                                       -args.r[station_ind, station_ind]))

        if sums[7] > 10:
            corr_path = '../pkl/corr_time' + str(args.case_num) + '.pkl'
            corr_time = pkl.load( open(corr_path, 'rb'))
            curr_corr = (sums[7]*sums[6]-sums[2]*sums[4])/(((sums[7]*sums[3]-sums[2]**2)**0.5)*((sums[7]*sums[5]-sums[4]**2)**0.5))
            print(curr_corr)
            corr_time.append(curr_corr)
            pkl.dump(corr_time, open(corr_path, 'wb'))

            waiting_path = '../pkl/waiting_time' + str(args.case_num) + '.pkl'
            curr_waiting = pkl.load(open(waiting_path, 'rb'))
            curr_waiting[0] = avg_time[0]
            curr_waiting[1] = avg_time[1]
            pkl.dump(curr_waiting, open(waiting_path, 'wb'))


    with server[station].request() as req:
        yield req


        # service time
        mu_ = mu[station, class_]
        ser_time = np.random.exponential(1 / mu_)

        yield env.timeout(ser_time)

        # with open('../pkl/avg_waiting'+str(case_num), 'rb') as f:
        #     avg_waiting = pkl.load(f)

        waiting_time = env.now - arrival_time


        name[station] += 1
        curr_waiting = (avg_time[station] * name[station]) / (name[station] + 1) + waiting_time / (name[station] + 1)
        # avg_waiting[station] = curr_waiting
        avg_time[station] = curr_waiting

        # with open('../pkl/avg_waiting'+str(case_num), 'wb') as f:
        #     pkl.dump(avg_waiting, f)


        # if customer is mismatched then she is redirected to the her designated queue
        if class_ != station:
             if station == 0:  # we redirect now only from station 0 now.
                station = class_

                arrival_time = env.now
                if args.is_corr:


                    if (sums[0] == 0) & (sums[1] == 0):
                        sums[0] = arrival_time
                    else:
                        if sums[1] > 0:  # cur_ind > 0:
                            prev = sums[1]
                            curr = arrival_time - sums[0]
                            sums[0] = arrival_time
                            sums[1] = curr
                            sums[2] += prev
                            sums[3] += prev**2
                            sums[4] += curr
                            sums[5] += curr**2
                            sums[6] += prev*curr
                            sums[7] += 1

                        else:
                            sums[1] = arrival_time - sums[0]
                            sums[0] = arrival_time

                env.process(service(env, name, server, mu, arrival_time, class_, station, size, True, case_num, args, sums, avg_time))



def customer_arrivals(env, server, r, mu, size, probabilities, ser_matched_rate, ser_mis_matched_rate, case_num, sums, avg_time):

    name = np.ones(size)*(-1)


    elements = list(np.arange(r.size))

    while True:

        # get the external stream identity
        arrival_identity = np.random.choice(elements, 1, p=probabilities)[0]

        class_ = int(np.remainder(arrival_identity, size))
        station = int(arrival_identity / size)

        curr_lamb = np.sum(r)
        yield env.timeout(np.random.exponential(1 / curr_lamb))

        arrival_time = env.now

        is_matched = station == class_
        env.process(service(env, name, server, mu, arrival_time, class_, station, size, is_matched,  case_num, args, sums, avg_time))




def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=np.array, help='external arrivals', default=np.array([]))
    parser.add_argument('--number_of_classes', type=int, help='number of classes', default=2)
    parser.add_argument('--mu', type=np.array, help='service rates', default=np.array([]))
    parser.add_argument('--end_time', type=float, help='The end of the simulation', default=7023000)
    parser.add_argument('--size', type=int, help='the number of stations in the system', default=2)
    parser.add_argument('--p_correct', type=float, help='the prob of external matched customer', default=0.5)
    parser.add_argument('--ser_matched_rate', type=float, help='service rate of matched customers', default=1.2)
    parser.add_argument('--ser_mis_matched_rate', type=float, help='service rate of mismatched customers', default=10.)
    parser.add_argument('--num_iterations', type=float, help='service rate of mismatched customers', default=10)
    parser.add_argument('--case_num', type=int, help='case number in my settings', default=random.randint(0, 100000))
    parser.add_argument('--df_summ', type=str, help='case number in my settings', default='../pkl/df_sum_res_sim_28.pkl')
    parser.add_argument('--is_corr', type=bool, help='should we keep track on inter departure', default=True)

    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

