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

    if sys.platform == 'linux':

        df = pd.read_excel('../files/exp_settings_comb.xlsx', sheet_name='Sheet1')

    else:
        df = pd.read_excel('../files/exp_settings_comb.xlsx', sheet_name='Sheet3')
        # df = pd.read_excel(r'G:\My Drive\Research\sum_results.xlsx', sheet_name='Sheet2')



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


        sums = [0,0,0,0,0,0,0,0]
        station_1_waits = []

        avg_time = list(np.zeros(args.size))

        args.waiting_pkl_path = '../pkl/waiting_time' + str(args.case_num) +'_'+ str(ind) +'.pkl'

        pkl.dump([0], open(args.waiting_pkl_path, 'wb'))

        b = 2.230380964364765
        mu = 1
        size = 1
        rho = 2/(b*mu)

        arrival_rate = 2 / b

        args.end_time = 100000000 / arrival_rate

        env.process(customer_arrivals(env, server, b, mu, size,  args.case_num, sums, avg_time, station_1_waits))
        env.run(until=(args.end_time))

        avg_waiting = pkl.load(open(args.waiting_pkl_path, 'rb'))

        total_avg_system = 0
        # for station_ind in range(args.size):
        #     df_summary_result.loc[ind, 'Arrival_'+str(station_ind)] = str(b)
        #     df_summary_result.loc[ind, 'Service_rate' + str(station_ind)] = str(mu)
        #     df_summary_result.loc[ind, 'avg_waiting_'+str(station_ind)] = avg_waiting[station_ind]
        #     df_summary_result.loc[ind, 'avg_sys_'+str(station_ind)] = avg_waiting[station_ind] * b
        #     total_avg_system += df_summary_result.loc[ind, 'avg_sys_'+str(station_ind)]
        #     df_summary_result.loc[ind, 'avg_sys_mm1_' + str(station_ind)] = rho / (1 - rho)
        #
        #
        # print(df_summary_result)

        print("--- %s seconds the %d th iteration ---" % (time.time() - start_time, 1))

        # with open('../pkl/df_summary_result_sim_different_sizes_queues_'+str(current_time)+'.pkl', 'wb') as f:
        #     pkl.dump(df_summary_result, f)
        # print('The average number of customers in station 1 is: ', df_summary_result.loc[0,'avg_sys_0'])


        if not os.path.exists(args.df_summ):
            df_ = pd.DataFrame([])
        else:
            df_ = pkl.load(open(args.df_summ, 'rb'))
        ind = df_.shape[0]

        df_.loc[ind, 'b'] = b
        df_.loc[ind, 'mu'] = mu
        df_.loc[ind, 'avg_cust_0'] = avg_waiting[0] * 2/b
        df_.loc[ind, 'avg_wait_0'] = avg_waiting[0]
        df_.loc[ind, 'sim_runtime'] = args.end_time

        pkl.dump(df_, open(args.df_summ,'wb'))

        print(df_)



def service(env, name, server, b, mu, arrival_time,  station,   case_num, args, sums, avg_time, station_1_waits):

    station_ind = 0
    arrival_rate = 2 / b



    # print(name[station])
    # print(int((args.end_time*0.4)*arrival_rate))

    # if (np.remainder(name[station], 50000) == 0):
    #     # wait_path = '../pkl/wait_station_1' + str(args.case_num) +'_'+'.pkl'
    #     # if int(name[station]/10000)>1:
    #     #     wait_90 = np.percentile(station_1_waits,90, axis=0)
    #     #     print('90 percentile is: ', wait_90)
    #     #     pkl.dump(wait_90, open(wait_path, 'wb'))
    #
    #     # print(len(station_1_waits))
    #     # print(b*np.array(station_1_waits).mean())
    #
    #     print('The current time is: ', env.now)
    #
    #     # arrival_rate = b
    #
    #     print('The average sys in station 0 is: ', avg_time[station_ind] * arrival_rate)
    #     print('The average waiting in station 0 is: ', avg_time[station_ind])
    #
    #     print(int(name[station]))
    #     print(int((args.end_time*0.4)*arrival_rate))
    #

    with server[0].request() as req:
        yield req

        # service time
        ser_time = np.random.exponential(1 / mu)

        yield env.timeout(ser_time)


        waiting_time = env.now - arrival_time
        if env.now > 10000:
            # station_1_waits.append(waiting_time)

            name[station] += 1
            curr_waiting = (avg_time[station] * name[station]) / (name[station] + 1) + waiting_time / (name[station] + 1)
            avg_time[station] = curr_waiting

            if  (int(name[station]) == int((args.end_time * 0.9) * arrival_rate)) or (int(name[station]) == int((args.end_time * 0.99) * arrival_rate)) or (
                    int(name[station]) == int((args.end_time * 0.999) * arrival_rate)):
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('Dumping avg_time')
                print(avg_time[0])
                pkl.dump(avg_time, open(args.waiting_pkl_path, 'wb'))



def customer_arrivals(env, server, b, mu, size,  case_num, sums, avg_time, station_1_waits):

    name = np.ones(size)*(-1)

    while True:


        yield env.timeout(np.random.uniform(0,b))
        # yield env.timeout(np.random.exponential(1 / b))

        arrival_time = env.now
        station = 0

        env.process(service(env, name, server, b, mu, arrival_time,  station,  case_num, args, sums, avg_time, station_1_waits))




def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=np.array, help='external arrivals', default=np.array([]))
    parser.add_argument('--number_of_classes', type=int, help='number of classes', default=2)
    parser.add_argument('--mu', type=np.array, help='service rates', default=np.array([]))
    parser.add_argument('--end_time', type=float, help='The end of the simulation', default=100050000)
    parser.add_argument('--size', type=int, help='the number of stations in the system', default=1)
    parser.add_argument('--p_correct', type=float, help='the prob of external matched customer', default=0.5)
    parser.add_argument('--ser_matched_rate', type=float, help='service rate of matched customers', default=1.2)
    parser.add_argument('--ser_mis_matched_rate', type=float, help='service rate of mismatched customers', default=10.)
    parser.add_argument('--num_iterations', type=float, help='service rate of mismatched customers', default=5)
    parser.add_argument('--case_num', type=int, help='case number in my settings', default=random.randint(0, 100000))
    parser.add_argument('--df_summ', type=str, help='case number in my settings', default='../pkl/df_sum_res_sim_gg1.pkl')
    parser.add_argument('--is_corr', type=bool, help='should we keep track on inter departure', default=True)
    parser.add_argument('--waiting_pkl_path', type=bool, help='the path of the average waiting time', default='../pkl/waiting_time')

    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

