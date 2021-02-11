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



def service(env, name, server, mu, arrival_time, class_,  size, pkl_name_inter_depart,
            pkl_name_inter_depart_mis ,pkl_enter_service_status):
    # if station == 1:
    #     pkl_name = os.path.join('inter_deparature_distribution', 'interdeparture_19_11_num_stations_d_'+str(size)+'.pkl')
    #     if not os.path.exists(pkl_name):
    #         df = pd.DataFrame([])
    #         with open(pkl_name, 'wb') as f:
    #             pkl.dump(df, f)
    #
    #     with open(pkl_name, 'rb') as f:
    #         df = pkl.load(f)
    #     ind = df.shape[0]
    #     df.loc[ind, 'departure_time'] = arrival_time
    #     if ind > 0:
    #         df.loc[ind, 'inter_departure'] = arrival_time - df.loc[ind - 1, 'departure_time']
    #     with open(pkl_name, 'wb') as f:
    #         pkl.dump(df, f)
    #     if np.remainder(name[station], 10000) == 0:
    #         print('The current time is: ', env.now)
    #
    #         with open('pkl/avg_waiting', 'rb') as f:
    #             avg_waiting = pkl.load(f)
    #         print(avg_waiting)
    #
    #         df_summary_result = pd.DataFrame([])
    #
    #         ind = df_summary_result.shape[0]
    #         total_avg_system = 0
    #         for station_ind in range(args.size):
    #             df_summary_result.loc[ind, 'Arrival_' + str(station_ind)] = str(args.r[station_ind])
    #             df_summary_result.loc[ind, 'avg_waiting_' + str(station_ind)] = avg_waiting[station_ind]
    #             df_summary_result.loc[ind, 'avg_sys_' + str(station_ind)] = avg_waiting[station_ind] * \
    #                                                                         (np.sum(args.r[station_ind, :]) +
    #                                                                          np.sum(args.r[:, station_ind])
    #                                                                          - args.r[station_ind, station_ind])
    #             total_avg_system += df_summary_result.loc[ind, 'avg_sys_' + str(station_ind)]
    #             # p0 = np.sum(args.r[:, station_ind]) / (
    #             #             np.sum(args.r[:, station_ind]) + np.sum(args.r[station_ind, :]) - args.r[
    #             #         station_ind, station_ind])
    #             # m = np.array([args.ser_matched_rate, args.ser_mis_matched_rate])
    #             # lamb = np.sum(args.r[:, station_ind]) + np.sum(args.r[station_ind, :]) - args.r[
    #             #     station_ind, station_ind]
    #             df_summary_result.loc[ind, 'avg_sys_mg1_' + str(station_ind)], rho = avg_sys(args.r, args.mu, station_ind)
    #             df_summary_result.loc[ind, 'avg_sys_mm1_' + str(station_ind)] = rho / (1 - rho)
    #
    #         df_summary_result.loc[ind, 'avg_sys_total'] = total_avg_system
    #
    #         now = datetime.now()
    #
    #         current_time = now.strftime("%H_%M_%S")
    #         with open('pkl/df_summary_result_sim_different_sizes_queues_' + str(current_time) + '.pkl', 'wb') as f:
    #             pkl.dump(df_summary_result, f)

    with server.request() as req:


        yield req

        with open(pkl_enter_service_status, 'rb') as f:
            df_enter_service_status = pkl.load(f)

        cur_ind = df_enter_service_status.shape[0]
        df_enter_service_status.loc[cur_ind, 'name'] = name
        df_enter_service_status.loc[cur_ind, 'time',] = env.now
        df_enter_service_status.loc[cur_ind, 'in_service'] = len(server.users)
        df_enter_service_status.loc[cur_ind, 'queue_length'] = len(server.queue)
        df_enter_service_status.loc[cur_ind, 'class'] = class_

        with open(pkl_enter_service_status, 'wb') as f:
            pkl.dump(df_enter_service_status, f)


        # service time

        ser_time = np.random.exponential(1 / mu[class_])

        yield env.timeout(ser_time)

        # with open('../inter_pkl/avg_waiting', 'rb') as f:
        #     avg_waiting = pkl.load(f)


        waiting_time = env.now - arrival_time
        # avg_waiting = (avg_waiting * name + waiting_time) / (name + 1)
        # with open('../inter_pkl/avg_waiting', 'wb') as f:
        #     pkl.dump(avg_waiting, f)

        # if class_ == 0:
        #     with open(pkl_name_inter_depart, 'rb') as f:
        #         dff = pkl.load(f)
        #
        #     ind = dff.shape[0]
        #     dff.loc[ind, 'Class'] = class_
        #     dff.loc[ind, 'Time'] = env.now
        #     dff.loc[ind, 'Waiting_time'] = waiting_time
        #     if ind > 1:
        #         dff.loc[ind, 'inter_departure'] = dff.loc[ind, 'Time']-dff.loc[ind-1, 'Time']
        #     with open(pkl_name_inter_depart, 'wb') as f:
        #         pkl.dump(dff, f)

        # if class_ == 1:
    with open(pkl_name_inter_depart_mis, 'rb') as f:
        dff = pkl.load(f)

    ind = dff.shape[0]
    dff.loc[ind, 'Class'] = class_
    dff.loc[ind, 'Time'] = env.now
    dff.loc[ind, 'Waiting_time'] = waiting_time
    dff.loc[ind, 'leaves_behind'] = len(server.users)+len(server.queue)

    if ind > 1:
        dff.loc[ind, 'inter_departure'] = dff.loc[ind, 'Time']-dff.loc[ind-1, 'Time']
    with open(pkl_name_inter_depart_mis, 'wb') as f:
        pkl.dump(dff, f)

def customer_arrivals(env, server, r, mu, size, pkl_name_inter_depart, pkl_name_inter_depart_mis, pkl_queue_arrival, pkl_enter_service_status):

    name = -1
    while True:

        # get the external stream identity
        p_0, p1 = r[0] / (np.sum(r)), r[1] / (np.sum(r))
        arrival_identity = np.random.choice(r.shape[0], 1, p=[p_0, p1])[0]
        class_ = arrival_identity


        curr_lamb = np.sum(r)
        yield env.timeout(np.random.exponential(1 / curr_lamb))

        arrival_time = env.now

        # update current customer

        name += 1

        with open(pkl_queue_arrival, 'rb') as f:
            df_queue_arrival = pkl.load(f)

        cur_ind = df_queue_arrival.shape[0]
        df_queue_arrival.loc[cur_ind, 'name'] = name
        df_queue_arrival.loc[cur_ind, 'time',] = env.now
        df_queue_arrival.loc[cur_ind, 'in_service'] = len(server.users)
        df_queue_arrival.loc[cur_ind, 'queue_length'] = len(server.queue)
        df_queue_arrival.loc[cur_ind, 'class'] = class_

        with open(pkl_queue_arrival, 'wb') as f:
            pkl.dump(df_queue_arrival, f)

        env.process(service(env, name, server, mu, arrival_time, class_,  size, pkl_name_inter_depart,
                            pkl_name_inter_depart_mis, pkl_enter_service_status))

def main(args):

    pkl_name_inter_depart = '../inter_pkl/inter_deparature_distribution.pkl'

    df = pd.DataFrame(columns = ['Class', 'Time', 'inter_departure', 'Waiting_time' ])
    with open(pkl_name_inter_depart, 'wb') as f:
        pkl.dump(df, f)

    pkl_name_inter_depart_mis = '../inter_pkl/inter_deparature_distribution_mis_05_service_1.pkl'
    df = pd.DataFrame(columns=['Class', 'Time', 'inter_departure', 'Waiting_time', 'leaves_behind'])
    with open(pkl_name_inter_depart_mis, 'wb') as f:
        pkl.dump(df, f)

    pkl_queue_arrival = '../inter_pkl/arrival_status.pkl'
    df_queue_arrival = pd.DataFrame(columns=['name', 'time', 'in_service', 'queue_length', 'class'])
    with open(pkl_queue_arrival, 'wb') as f:
        pkl.dump(df_queue_arrival, f)

    pkl_enter_service_status = '../inter_pkl/enter_service_status.pkl'
    df_pkl_enter_service_status = pd.DataFrame(columns=['name', 'time', 'in_service', 'queue_length', 'class'])
    with open(pkl_enter_service_status, 'wb') as f:
        pkl.dump(df_pkl_enter_service_status, f)

    df_summary_result = pd.DataFrame([])
    start_time = time.time()
    env = simpy.Environment()



    server = simpy.Resource(env, capacity=args.num_servers)

    # avg_waiting = 0
    # with open('../inter_pkl/avg_waiting', 'wb') as f:
    #     pkl.dump(avg_waiting, f)

    env.process(customer_arrivals(env, server, args.r, args.mu, args.size, pkl_name_inter_depart,
                                  pkl_name_inter_depart_mis, pkl_queue_arrival, pkl_enter_service_status))
    env.run(until=(args.end_time))

    # with open('../inter_pkl/avg_waiting', 'rb') as f:
    #     avg_waiting = pkl.load(f)
    # print(avg_waiting)

    ind = df_summary_result.shape[0]
    total_avg_system = 0


    print("--- %s seconds the %d th iteration ---" % (time.time() - start_time, 1))
    now = datetime.now()

    current_time = now.strftime("%H_%M_%S")
    # with open('pkl/df_summary_result_sim_different_sizes_queues_'+str(current_time)+'.pkl', 'wb') as f:
    #     pkl.dump(df_summary_result, f)

def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=np.array, help='external arrivals', default=np.array([0.5,0.5]))
    parser.add_argument('--number_of_classes', type=int, help='number of classes', default=2)
    parser.add_argument('--mu', type=np.array, help='service rates', default=np.array([1.0, 5000000]))
    parser.add_argument('--end_time', type=float, help='The end of the simulation', default=20000)
    parser.add_argument('--size', type=int, help='the number of stations in the queue', default=1)
    parser.add_argument('--p_correct', type=float, help='the prob of external matched customer', default=0.5)
    parser.add_argument('--ser_matched_rate', type=float, help='service rate of matched customers', default=1.5)
    parser.add_argument('--ser_mis_matched_rate', type=float, help='service rate of mismatched customers', default=50000.)
    parser.add_argument('--num_servers', type=int, help='number of servers in station', default=1)



    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    print('git check')
    main(args)

