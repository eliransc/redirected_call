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
import pickle as pkl
import matplotlib.pyplot as plt
import sympy
from sympy import *
from scipy.special import gamma, factorial


def gamma_pdf(x, theta, k):
    return (1 / (gamma(k))) * (1 / theta ** k) * (np.exp(-x / theta))


def gamma_lst(s, theta, k):
    return (1 + theta * s) ** (-k)


def get_nth_moment(theta, k, n):
    s = Symbol('s')
    y = gamma_lst(s, theta, k)
    for i in range(n):
        if i == 0:
            dx = diff(y, s)
        else:
            dx = diff(dx, s)
    return ((-1) ** n) * dx.subs(s, 0)


def gamma_lst(s, theta, k):
    return (1 + theta * s) ** (-k)


def unif_lst(s, b, a=0):
    return (1 / (b - a)) * ((np.exp(-a * s) - np.exp(-b * s)) / s)


def n_mom_uniform(n, b, a=0):
    return (1 / ((n + 1) * (b - a))) * (b ** (n + 1) - a ** (n + 1))


def laplace_mgf(t, mu, b):
    return exp(mu * t) / (1 - (b ** 2) * (t ** 2))


def nthmomlap(mu, b, n):
    t = Symbol('t')
    y = laplace_mgf(t, mu, b)
    for i in range(n):
        if i == 0:
            dx = diff(y, t)
        else:
            dx = diff(dx, t)
    return dx.subs(t, 0)


def normal_mgf(t, mu, sig):
    return exp(mu * t + (sig ** 2) * (t ** 2) / 2)


def nthmomnormal(mu, sig, n):
    t = Symbol('t')
    y = normal_mgf(t, mu, sig)
    for i in range(n):
        if i == 0:
            dx = diff(y, t)
        else:
            dx = diff(dx, t)
    return dx.subs(t, 0)


def generate_unif(is_arrival):
    if is_arrival:
        b_arrive = np.random.uniform(2, 10)
        a_arrive = 0
        moms_arr = []
        for n in range(1, 11):
            moms_arr.append(n_mom_uniform(n, b_arrive))
        return (a_arrive, b_arrive, moms_arr)
    else:
        b_ser = 2
        a_ser = 0
        moms_ser = []
        for n in range(1, 11):
            moms_ser.append(n_mom_uniform(n, b_ser))
        return (a_ser, b_ser, moms_ser)


def generate_gamma(is_arrival):
    if is_arrival:
        rho = np.random.uniform(0.1, 0.99)
        shape = np.random.uniform(0.1, 100)
        scale = 1 / (rho * shape)
        moms_arr = np.array([])
        for mom in range(1, 11):
            moms_arr = np.append(moms_arr, np.array(N(get_nth_moment(shape, scale, mom))).astype(np.float64))
        return (shape, scale, moms_arr)
    else:
        shape = np.random.uniform(1, 100)
        scale = 1 / shape
        moms_ser = np.array([])
        for mom in range(1, 11):
            moms_ser = np.append(moms_ser, np.array(N(get_nth_moment(shape, scale, mom))).astype(np.float64))
        return (shape, scale, moms_ser)


def generate_normal(is_arrival):
    if is_arrival:
        mu = np.random.uniform(1.5, 10)
        sig = np.random.uniform(mu / 6, mu / 4)

        moms_arr = np.array([])
        for mom in tqdm(range(1, 11)):
            moms_arr = np.append(moms_arr, np.array(N(nthmomnormal(mu, sig, mom))).astype(np.float64))

        return (mu, sig, moms_arr)
    else:
        mu = 1
        sig = np.random.uniform(0.15, 0.22)

        moms_ser = np.array([])
        for mom in tqdm(range(1, 11)):
            moms_ser = np.append(moms_ser, np.array(N(nthmomnormal(mu, sig, mom))).astype(np.float64))
        return (mu, sig, moms_ser)

def main(args):

    # if sys.platform == 'linux':
    #
    #     df = pd.read_excel('../files/exp_settings_comb.xlsx', sheet_name='Sheet1')
    #
    # else:
    #     df = pd.read_excel('../files/exp_settings_comb.xlsx', sheet_name='Sheet3')
    #     # df = pd.read_excel(r'G:\My Drive\Research\sum_results.xlsx', sheet_name='Sheet2')

    arrival_dist = np.random.choice([1, 2, 3], size=1, replace=True, p=[0.3, 0.4, 0.3])[0]
    ser_dist = np.random.choice([1, 2, 3], size=1, replace=True, p=[0.3, 0.4, 0.3])[0]

    if arrival_dist == 1:
        arrival_dist_params = generate_unif(True)
    elif arrival_dist == 2:
        arrival_dist_params = generate_gamma(True)
    else:
        arrival_dist_params = generate_normal(True)

    if ser_dist == 1:
        ser_dist_params = generate_unif(False)
    elif ser_dist == 2:
        ser_dist_params = generate_gamma(False)
    else:
        ser_dist_params = generate_normal(False)

    print(arrival_dist_params, ser_dist_params, arrival_dist, ser_dist)

    arrival_rate = 1/arrival_dist_params[2][0]

    waiting_time_list = []
    pkl.dump(waiting_time_list, open('../pkl/waiting_time_station_1_'+str(args.case_num)+'.pkl', 'wb'))

    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")

    df_summary_result = pd.DataFrame([])

    for ind in range(args.num_iterations):

        start_time = time.time()

        env = simpy.Environment()

        server = []
        for server_ind in range(args.size):
            server.append(simpy.Resource(env, capacity=1))



        sums = [0,0,0,0,0,0,0,0]
        station_1_waits = []



        avg_time = list(np.zeros(args.size))

        args.waiting_pkl_path = '../pkl/waiting_time' + str(args.case_num) + '.pkl'

        pkl.dump([0], open(args.waiting_pkl_path, 'wb'))

        # b = 2.230380964364765
        # mu = 1
        size = 1
        # rho = 2/(b*mu)

        # arrival_rate = 2 / b  # 1/arrival_dist_params[2][0]

        args.end_time = 100000000 / arrival_rate

        env.process(customer_arrivals(env, server, size,  args.case_num, sums, avg_time, station_1_waits, arrival_dist_params, ser_dist_params, arrival_dist, ser_dist, arrival_rate))
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

        with open('../pkl/df_summary_result_sim_different_sizes_queues_'+str(current_time)+'.pkl', 'wb') as f:
            pkl.dump(df_summary_result, f)
        print('The average number of customers in station 1 is: ', df_summary_result.loc[0,'avg_sys_0'])


        if not os.path.exists(args.df_summ):
            df_ = pd.DataFrame([])
        else:
            df_ = pkl.load(open(args.df_summ, 'rb'))
        ind = df_.shape[0]

        if arrival_dist == 1:
            df_.loc[ind, 'arrival_dist']  = 'Uniform'
            df_.loc[ind, 'arrival_params'] = str(arrival_dist_params[0])+'_'+ str(arrival_dist_params[1])
        elif arrival_dist == 2:
            df_.loc[ind, 'arrival_dist']  = 'Gamma'
            df_.loc[ind, 'arrival_params'] = str(arrival_dist_params[0])+'_'+ str(arrival_dist_params[1])
        else:
            df_.loc[ind, 'arrival_dist']  = 'Normal'
            df_.loc[ind, 'arrival_params'] = str(arrival_dist_params[0])+'_'+ str(arrival_dist_params[1])


        if ser_dist == 1:
            df_.loc[ind, 'ser_dist']  = 'Uniform'
            df_.loc[ind, 'ser_params'] = str(arrival_dist_params[0])+'_'+ str(arrival_dist_params[1])
        elif ser_dist == 2:
            df_.loc[ind, 'ser_dist']  = 'Gamma'
            df_.loc[ind, 'ser_params'] = str(arrival_dist_params[0])+'_'+ str(arrival_dist_params[1])
        else:
            df_.loc[ind, 'ser_dist']  = 'Normal'
            df_.loc[ind, 'ser_params'] = str(arrival_dist_params[0])+'_'+ str(arrival_dist_params[1])


        df_.loc[ind, 'avg_cust_0'] = avg_waiting[0] * arrival_rate
        df_.loc[ind, 'avg_wait_0'] = avg_waiting[0]
        df_.loc[ind, 'sim_runtime'] = args.end_time



        pkl.dump(df_, open(args.df_summ,'wb'))

        print(df_)



def service(env, name, server, arrival_time,  station,   case_num, args, sums, avg_time, station_1_waits, arrival_dist_params, ser_dist_params, arrival_dist, ser_dist, arrival_rate):

    station_ind = 0

    if (np.remainder(name[station], 500000) == 0):
        # wait_path = '../pkl/wait_station_1' + str(args.case_num) +'_'+'.pkl'
        # if int(name[station]/10000)>1:
        #     wait_90 = np.percentile(station_1_waits,90, axis=0)
        #     print('90 percentile is: ', wait_90)
        #     pkl.dump(wait_90, open(wait_path, 'wb'))

        # print(len(station_1_waits))
        # print(b*np.array(station_1_waits).mean())

        print('The current time is: ', env.now)

        # arrival_rate = b

        print('The average sys in station 0 is: ', avg_time[station_ind] * arrival_rate)
        print('The average waiting in station 0 is: ', avg_time[station_ind])

        print(int(name[station]))
        print(int((args.end_time*0.99)*arrival_rate))



    if  int(name[station]) == int((args.end_time*0.99)*arrival_rate):

        print('Dumping avg_time')
        pkl.dump(avg_time, open(args.waiting_pkl_path, 'wb'))



    with server[0].request() as req:
        yield req

        # service time

        if arrival_dist == 1:
            a_arrive = arrival_dist_params[0]
            b_arrive = arrival_dist_params[1]

            yield env.timeout(np.random.uniform(a_arrive, b_arrive))


        elif arrival_dist == 2:
            shape = arrival_dist_params[0]
            scale = arrival_dist_params[1]
            yield env.timeout(np.random.gamma(shape, scale))
        else:
            mu = arrival_dist_params[0]
            sig = arrival_dist_params[1]
            yield env.timeout(np.random.normal(mu, sig))



        waiting_time = env.now - arrival_time
        if env.now > 10000:
            # station_1_waits.append(waiting_time)

            name[station] += 1
            curr_waiting = (avg_time[station] * name[station]) / (name[station] + 1) + waiting_time / (name[station] + 1)
            avg_time[station] = curr_waiting



def customer_arrivals(env, server,  size,  case_num, sums, avg_time, station_1_waits, arrival_dist_params, ser_dist_params, arrival_dist, ser_dist, arrival_rate):

    name = np.ones(size)*(-1)

    while True:

        if arrival_dist == 1:
            a_arrive = arrival_dist_params[0]
            b_arrive = arrival_dist_params[1]

            yield env.timeout(np.random.uniform(a_arrive, b_arrive))


        elif arrival_dist == 2:
            shape = arrival_dist_params[0]
            scale = arrival_dist_params[1]
            yield env.timeout(np.random.gamma(shape, scale))
        else:
            mu = arrival_dist_params[0]
            sig = arrival_dist_params[1]
            yield env.timeout( np.random.normal(mu, sig))


        arrival_time = env.now
        station = 0

        env.process(service(env, name, server,  arrival_time,  station,  case_num, args, sums, avg_time, station_1_waits, arrival_dist_params, ser_dist_params, arrival_dist, ser_dist, arrival_rate))




def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=np.array, help='external arrivals', default=np.array([]))
    parser.add_argument('--number_of_classes', type=int, help='number of classes', default=2)
    parser.add_argument('--mu', type=np.array, help='service rates', default=np.array([]))
    parser.add_argument('--end_time', type=float, help='The end of the simulation', default=100000000)
    parser.add_argument('--size', type=int, help='the number of stations in the system', default=1)
    parser.add_argument('--p_correct', type=float, help='the prob of external matched customer', default=0.5)
    parser.add_argument('--ser_matched_rate', type=float, help='service rate of matched customers', default=1.2)
    parser.add_argument('--ser_mis_matched_rate', type=float, help='service rate of mismatched customers', default=10.)
    parser.add_argument('--num_iterations', type=float, help='service rate of mismatched customers', default=1)
    parser.add_argument('--case_num', type=int, help='case number in my settings', default=random.randint(0, 100000))
    parser.add_argument('--df_summ', type=str, help='case number in my settings', default='../pkl/df_sum_res_sim_gg1.pkl')
    parser.add_argument('--is_corr', type=bool, help='should we keep track on inter departure', default=True)
    parser.add_argument('--waiting_pkl_path', type=bool, help='the path of the average waiting time', default='../pkl/waiting_time')

    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

