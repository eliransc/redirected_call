import simpy
import numpy as np
import sys
import argparse
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import time
import os

def service_0(env, name, server, mu, arrival_time, arrival_identity):




    with server[0].request() as req:
        yield req

        # service time
        mu_ = mu.flatten()[arrival_identity]
        ser_time = np.random.exponential(1 / mu_)

        yield env.timeout(ser_time)

        with open('avg_waiting', 'rb') as f:
            avg_waiting = pkl.load(f)

        # if arrival_identity == 1: # we want interdeparture time of only misclassified customers in station 0
        #     pkl_name = os.path.join('inter_deparature_distribution', 'interdeparture_5_11_b.pkl')
        #     if not os.path.exists(pkl_name):
        #         df = pd.DataFrame([])
        #         with open(pkl_name, 'wb') as f:
        #             pkl.dump(df, f)
        #
        #     with open(pkl_name, 'rb') as f:
        #         df = pkl.load(f)
        #     ind = df.shape[0]
        #     departure_time = env.now
        #     df.loc[ind, 'departure_time'] = departure_time
        #     if ind > 0:
        #         df.loc[ind, 'inter_departure'] = departure_time - df.loc[ind - 1, 'departure_time']
        #     with open(pkl_name, 'wb') as f:
        #         pkl.dump(df, f)

        waiting_time = env.now - arrival_time
        avg_waiting[0] = (avg_waiting[0] * name[0] + waiting_time) / (name[0] + 1)
        with open('avg_waiting', 'wb') as f:
            pkl.dump(avg_waiting, f)
        # if customer arrival as type 0 then redirect to the other queue
        if arrival_identity == 1:
            arrival_identity = 3
            name[1] += 1
            arrival_time = env.now
            env.process(service_1(env, name, server, mu, arrival_time, arrival_identity))


def service_1(env, name, server, mu, arrival_time, arrival_identity):


    with server[1].request() as req:
        yield req

        # service time
        mu_ = mu.flatten()[arrival_identity]
        ser_time = np.random.exponential(1 / mu_)

        yield env.timeout(ser_time)

        with open('avg_waiting', 'rb') as f:
            avg_waiting = pkl.load(f)


        waiting_time = env.now - arrival_time
        avg_waiting[1] = (avg_waiting[1] * name[1] + waiting_time) / (name[1] + 1).astype(float)
        with open('avg_waiting', 'wb') as f:
            pkl.dump(avg_waiting, f)
        # if customer arrival as type 0 then redirect to the other queue

    if arrival_identity == 2:
        arrival_identity = 0
        name[0] += 1
        arrival_time = env.now

        pkl_name = os.path.join('inter_deparature_distribution', 'interdeparture_class1_08_02_service15_and2_only_misclassed.pkl')
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

        env.process(service_0(env, name, server, mu, arrival_time, arrival_identity))


def customer_arrivals(env, server, r, mu):

    name = np.array([-1, -1])

    while True:
        # get the external stream identity
        elements = [0, 1, 2, 3]
        probabilities = ((r) / np.sum(r)).flatten()
        arrival_identity = np.random.choice(elements, 1, p=probabilities)[0]

        curr_lamb = np.sum(r)
        yield env.timeout(np.random.exponential(1 / curr_lamb))

        arrival_time = env.now

        # update current customer
        if arrival_identity <= 1:
            name[0] += 1
            env.process(service_0(env, name, server, mu, arrival_time, arrival_identity))
        else:
            name[1] += 1
            env.process(service_1(env, name, server, mu, arrival_time, arrival_identity))



def main(args):



    df_summary_result = pd.DataFrame([], columns=['Arrival_0', 'Arrival_1'])

    print(args.r)
    start_time = time.time()


    df_queue = pd.DataFrame([], columns=['time', 'event', 'num_of_0', 'num_of_1', 'time_between'])
    df_queue.loc[0, 'time'] = 0
    df_queue.loc[0, 'event'] = 'begin'
    df_queue.loc[0, 'num_of_0'] = 0
    df_queue.loc[0, 'num_of_1'] = 0
    df_queue.loc[0, 'time_between'] = 0


    avg_waiting_0 = 0.
    avg_waiting_1 = 0.
    with open('avg_waiting', 'wb') as f:
        pkl.dump([avg_waiting_0, avg_waiting_1], f)
    env = simpy.Environment()
    server_0 = simpy.Resource(env, capacity=1)
    server_1 = simpy.Resource(env, capacity=1)
    server = [server_0, server_1]
    env.process(customer_arrivals(env, server, args.r, args.mu))
    env.run(until=args.end_time)

    with open('avg_waiting', 'rb') as f:
        avg_waiting = pkl.load(f)
    print(avg_waiting)

    ind = df_summary_result.shape[0]
    df_summary_result.loc[ind, 'Arrival_0'] = args.r[0]
    df_summary_result.loc[ind, 'Arrival_1'] = args.r[1]
    df_summary_result.loc[ind, 'avg_waiting_0'] = avg_waiting[0]
    df_summary_result.loc[ind, 'avg_waiting_1'] = avg_waiting[1]
    df_summary_result.loc[ind, 'avg_sys_0'] = avg_waiting[0]*(args.r[0,0]+args.r[1,0]+args.r[0,1])
    df_summary_result.loc[ind, 'avg_sys_1'] = avg_waiting[1]*(args.r[1,1]+args.r[1,0]+args.r[0,1])

    print("--- %s seconds the %d th iteration ---" % (time.time() - start_time, 1))

    with open('df_summary_result_sim_3_11_inter.pkl', 'wb') as f:
        pkl.dump(df_summary_result, f)

    print(df_summary_result)

def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=np.array, help='external arrivals', default=np.array([[0.7, 0.3], [0.3, 0.7]]))
    parser.add_argument('--number_of_classes', type=int, help='number of classes', default=2)
    parser.add_argument('--mu', type=np.array, help='service rates', default=np.array([[1.5, 2.], [2., 1.5]]))
    parser.add_argument('--end_time', type=float, help='The end of the simulation', default=180000)
    parser.add_argument('--n_max', type=int, help='numerical_limit for steady-state', default=100)

    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

