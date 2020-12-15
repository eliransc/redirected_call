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



def service(env, name, server, mu, arrival_time, class_, station, size):
    if station == 1:
        pkl_name = os.path.join('inter_deparature_distribution', 'interdeparture_19_11_num_stations_d_'+str(size)+'.pkl')
        if not os.path.exists(pkl_name):
            df = pd.DataFrame([])
            with open(pkl_name, 'wb') as f:
                pkl.dump(df, f)

        with open(pkl_name, 'rb') as f:
            df = pkl.load(f)
        ind = df.shape[0]
        df.loc[ind, 'departure_time'] = arrival_time
        if ind > 0:
            df.loc[ind, 'inter_departure'] = arrival_time - df.loc[ind - 1, 'departure_time']
        with open(pkl_name, 'wb') as f:
            pkl.dump(df, f)
        if np.remainder(name[station], 10000) == 0:
            print('The current time is: ', env.now)

            with open('pkl/avg_waiting', 'rb') as f:
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
                # p0 = np.sum(args.r[:, station_ind]) / (
                #             np.sum(args.r[:, station_ind]) + np.sum(args.r[station_ind, :]) - args.r[
                #         station_ind, station_ind])
                # m = np.array([args.ser_matched_rate, args.ser_mis_matched_rate])
                # lamb = np.sum(args.r[:, station_ind]) + np.sum(args.r[station_ind, :]) - args.r[
                #     station_ind, station_ind]
                df_summary_result.loc[ind, 'avg_sys_mg1_' + str(station_ind)], rho = avg_sys(args.r, args.mu, station_ind)
                df_summary_result.loc[ind, 'avg_sys_mm1_' + str(station_ind)] = rho / (1 - rho)

            df_summary_result.loc[ind, 'avg_sys_total'] = total_avg_system

            now = datetime.now()

            current_time = now.strftime("%H_%M_%S")
            with open('pkl/df_summary_result_sim_different_sizes_queues_' + str(current_time) + '.pkl', 'wb') as f:
                pkl.dump(df_summary_result, f)

    with server[station].request() as req:
        yield req

        # service time
        mu_ = mu[station, class_]
        ser_time = np.random.exponential(1 / mu_)

        yield env.timeout(ser_time)

        with open('pkl/avg_waiting', 'rb') as f:
            avg_waiting = pkl.load(f)

        waiting_time = env.now - arrival_time
        avg_waiting[station] = (avg_waiting[station] * name[station] + waiting_time) / (name[station] + 1)
        with open('pkl/avg_waiting', 'wb') as f:
            pkl.dump(avg_waiting, f)
        # if customer is mismatched then she is redirected to the her designated queue
        if class_ != station:
            station = class_
            name[station] += 1
            arrival_time = env.now
            env.process(service(env, name, server, mu, arrival_time, class_, station, size))



def customer_arrivals(env, server, r, mu, size, probabilities, ser_matched_rate, ser_mis_matched_rate):

    name = np.ones(size)*(-1)

    effective_rates = np.zeros(size)
    avg_service = np.zeros(size)
    # for ind in range(size):
    #     effective_rates[ind] = np.sum(r[ind,:]) + np.sum(r[:, ind]) - r[ind, ind]
    #     avg_service[ind] = (r[ind, :]/effective_rates[ind])*(1/ser_matched_rate)\
    #                        +(1-(r[ind,ind])/effective_rates[ind])*(1/ser_mis_matched_rate)
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
        env.process(service(env, name, server, mu, arrival_time, class_, station, size))

def main(args):


    df_summary_result = pd.DataFrame([])
    start_time = time.time()




    env = simpy.Environment()

    server = []
    for server_ind in range(args.size):
        server.append(simpy.Resource(env, capacity=args.num_servers))
    p_incorrect = (1 - args.p_correct) / (args.size - 1)
    args.r = np.identity(args.size)
    args.r = np.where(args.r == 1, args.p_correct, p_incorrect)



    env.process(customer_arrivals(env, server, args.r, args.mu, args.size))
    env.run(until=(args.end_time))

    with open('pkl/avg_waiting', 'rb') as f:
        avg_waiting = pkl.load(f)
    print(avg_waiting)

    ind = df_summary_result.shape[0]
    total_avg_system = 0
    for station_ind in range(args.size):
        df_summary_result.loc[ind, 'Arrival_'+str(station_ind)] = str(args.r[station_ind])
        df_summary_result.loc[ind, 'avg_waiting_'+str(station_ind)] = avg_waiting[station_ind]
        df_summary_result.loc[ind, 'avg_sys_'+str(station_ind)] = avg_waiting[station_ind] *\
                                                                  (np.sum(args.r[station_ind,:]) +
                                                                   np.sum(args.r[:, station_ind])
                                                                   -args.r[station_ind, station_ind])
        total_avg_system +=  df_summary_result.loc[ind, 'avg_sys_'+str(station_ind)]
        p0 = np.sum(args.r[:,station_ind])/(np.sum(args.r[:,station_ind])+np.sum(args.r[station_ind,:])-args.r[station_ind,station_ind])
        m = np.array([args.ser_matched_rate, args.ser_mis_matched_rate])
        lamb = np.sum(args.r[:,station_ind])+np.sum(args.r[station_ind,:])-args.r[station_ind,station_ind]
        df_summary_result.loc[ind, 'avg_sys_mg1_'+str(station_ind)], rho = avg_sys(args.r, args.mu, station_ind)
        df_summary_result.loc[ind, 'avg_sys_mm1_' + str(station_ind)] = rho/(1-rho)

        df_summary_result.loc[ind, 'avg_sys_total'] = total_avg_system

        print("--- %s seconds the %d th iteration ---" % (time.time() - start_time, 1))
        now = datetime.now()

        current_time = now.strftime("%H_%M_%S")
        with open('pkl/df_summary_result_sim_different_sizes_queues_'+str(current_time)+'.pkl', 'wb') as f:
            pkl.dump(df_summary_result, f)

    print(df_summary_result)

def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=np.array, help='external arrivals', default=np.array([]))
    parser.add_argument('--number_of_classes', type=int, help='number of classes', default=2)
    parser.add_argument('--mu', type=np.array, help='service rates', default=np.array([]))
    parser.add_argument('--end_time', type=float, help='The end of the simulation', default=250000)
    parser.add_argument('--size', type=int, help='the number of stations in the queue', default=1)
    parser.add_argument('--p_correct', type=float, help='the prob of external matched customer', default=0.5)
    parser.add_argument('--ser_matched_rate', type=float, help='service rate of matched customers', default=4)
    parser.add_argument('--ser_mis_matched_rate', type=float, help='service rate of mismatched customers', default=1.6)
    parser.add_argument('--num_servers', type=int, help='number of servers in station', default=1)



    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    print('git check')
    main(args)

