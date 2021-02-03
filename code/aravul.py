# imports
import simpy
import numpy as np
import sys
import argparse
import pickle as pkl
from tqdm import tqdm
import pandas as pd


def service_b(env, server_a, server_b, mu_a, mu_b, arrival_time, name,sigma_a, sigma_b, args, sim_num):

    with server_b.request() as req:
        yield req

        with open('../pkl/server_b_available.pkl', 'rb') as f:
            server_options = pkl.load(f)
        # print(server_options)

        rand_pick_index = np.random.randint(server_options.shape[0], size=1)[0]
        pick_server = server_options[rand_pick_index]
        server_options = np.delete(server_options, rand_pick_index)

        with open('../pkl/server_b_available.pkl', 'wb') as f:
            pkl.dump(server_options, f)

        # service time
        ser_time = np.max(np.random.normal(mu_b[pick_server], sigma_b, 1), 0)

        with open('../pkl/number_of_ready_stage_0_df.pkl', 'rb') as f:
            number_of_ready_stage_0_df = pkl.load(f)

        ind = number_of_ready_stage_0_df.shape[0]
        number_of_ready_stage_0_df.loc[ind, 'num_ready'] = number_of_ready_stage_0_df.loc[ind-1, 'num_ready']-1
        number_of_ready_stage_0_df.loc[ind, 'time'] = env.now +sim_num*args.end_time

        with open('../pkl/number_of_ready_stage_0_df.pkl', 'wb') as f:
            pkl.dump(number_of_ready_stage_0_df, f)

        if env.now+ser_time < args.end_time:
            yield env.timeout(ser_time)
        else:
            yield env.timeout(args.end_time-env.now)


        with open('../pkl/server_b_available.pkl', 'rb') as f:
            server_options = pkl.load(f)

        server_options = np.append(server_options, pick_server)
        # print(server_options)

        with open('../pkl/server_b_available.pkl', 'wb') as f:
            pkl.dump(server_options, f)

        with open('../pkl/caseta_counter.pkl', 'rb') as f:
            caseta_counter = pkl.load(f)

        caseta_counter += 1

        with open('../pkl/caseta_counter.pkl', 'wb') as f:
            pkl.dump(caseta_counter, f)

    env.process(service_a(env, server_a, server_b, mu_a, mu_b, arrival_time, name, sigma_a, sigma_b, args, sim_num))

def service_a(env, server_a, server_b, mu_a, mu_b, arrival_time, name, sigma_a, sigma_b, args, sim_num):

    if env.now > 7.2:
        print('ee')

    with server_a.request() as req:

        yield req

        with open('../pkl/server_a_available.pkl', 'rb') as f:
            server_options = pkl.load(f)
        # print(server_options)

        rand_pick_index = np.random.randint(server_options.shape[0], size=1)[0]
        pick_server = server_options[rand_pick_index]
        server_options = np.delete(server_options, rand_pick_index)

        with open('../pkl/server_a_available.pkl', 'wb') as f:
            pkl.dump(server_options, f)

        # service time
        ser_time = np.max(np.random.normal(mu_a[pick_server], sigma_a, 1), 0) # compute service time according to the worker

        with open('../pkl/number_of_ready_stage_0_df.pkl', 'rb') as f:
            number_of_ready_stage_0_df = pkl.load(f)

        if env.now+ser_time < args.end_time:
            yield env.timeout(ser_time)
        else:
            yield env.timeout(args.end_time-env.now)

        ind = number_of_ready_stage_0_df.shape[0] # find the current ind in the dataframe
        number_of_ready_stage_0_df.loc[ind, 'num_ready'] = number_of_ready_stage_0_df.loc[ind-1, 'num_ready']+1
        number_of_ready_stage_0_df.loc[ind, 'time'] = env.now+sim_num*args.end_time

        with open('../pkl/number_of_ready_stage_0_df.pkl', 'wb') as f: # update the DF
            pkl.dump(number_of_ready_stage_0_df, f)

        with open('../pkl/server_a_available.pkl', 'rb') as f:
            server_options = pkl.load(f)

        server_options = np.append(server_options, pick_server)
        # print(server_options)

        with open('../pkl/server_a_available.pkl', 'wb') as f:
            pkl.dump(server_options, f)


    env.process(service_b(env, server_a, server_b, mu_a, mu_b, arrival_time, name, sigma_a, sigma_b, args, sim_num))

def customer_arrivals(env, server_a, server_b, mu_a, mu_b, total_caseta, end_time, sigma_a, sigma_b, args, sim_num):

    name = 0


    while True:

        # get the external stream identity
        if name < total_caseta:

            yield env.timeout(0.0001)
        else:
            yield env.timeout(end_time)

        arrival_time = env.now

        # update current customer

        name += 1
        env.process(service_a(env, server_a, server_b, mu_a, mu_b, arrival_time, name, sigma_a, sigma_b, args, sim_num))


def main(args):

    throughput = []
    for num_caseta in tqdm(range(25, 40)):

        number_of_ready_stage_0_df = pd.DataFrame([])
        number_of_ready_stage_0_df.loc[0,'num_ready'] = 0
        number_of_ready_stage_0_df.loc[0, 'time'] = 0
        with open('../pkl/number_of_ready_stage_0_df.pkl', 'wb') as f:
            pkl.dump(number_of_ready_stage_0_df, f)

        caseta_counter = 0

        with open('../pkl/caseta_counter.pkl', 'wb') as f:
            pkl.dump(caseta_counter, f)

        for sim_num in range(args.num_days):
            with open('../pkl/server_a_available.pkl', 'wb') as f:
                pkl.dump(np.arange(args.num_servers_a), f)

            with open('../pkl/server_b_available.pkl', 'wb') as f:
                pkl.dump(np.arange(args.num_servers_b), f)


            print(sim_num)
            env = simpy.Environment()

            server_a = simpy.Resource(env, capacity=args.num_servers_a)
            server_b = simpy.Resource(env, capacity=args.num_servers_b)

            env.process(customer_arrivals(env, server_a, server_b,   args.mu_a, args.mu_b, num_caseta,  args.end_time,
                                          args.sigma_a, args.sigma_b, args, sim_num))
            env.run(until=(args.end_time))

        with open('../pkl/caseta_counter.pkl', 'rb') as f:
            caseta_counter = pkl.load(f)
        print(caseta_counter / args.end_time)
        throughput.append(caseta_counter / args.end_time)

    print(throughput)
    with open('../pkl/throughput.pkl', 'wb') as f:
        pkl.dump(throughput, f)


def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--end_time', type=int, help='end time in the sim', default=8)
    parser.add_argument('--mu_a', type=np.array, help='service rate', default=np.array([1.79, 1.59, 1.33, 1.34, 1.26]))
    parser.add_argument('--mu_b', type=np.array, help='service rate', default=np.array([3.73, 3.27, 2.86, 2.79, 2.74, 2.56,2.51,2.49,2.48,2.55,2.49]))
    parser.add_argument('--sigma_a', type=float, help='service rate', default=0.1)
    parser.add_argument('--sigma_b', type=float, help='service rate', default=0.2)
    parser.add_argument('--total_caseta', type=float, help='total number of casetas', default=5)
    parser.add_argument('--num_servers_a', type=int, help='total number of casetas', default=5)
    parser.add_argument('--num_servers_b', type=int, help='total number of casetas', default=11)
    parser.add_argument('--num_days', type=int, help='total number of casetas', default=50)
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
