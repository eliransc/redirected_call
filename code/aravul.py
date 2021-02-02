# imports
import simpy
import numpy as np
import sys
import argparse
import pickle as pkl
from tqdm import tqdm

def service_b(env, server_a, server_b, mu_a, mu_b, arrival_time, name,sigma_a, sigma_b):

    with server_b.request() as req:
        yield req

        ser_time = np.max(np.random.normal(mu_b, sigma_b, 1), 0)
        yield env.timeout(ser_time)

        with open('../pkl/caseta_counter.pkl', 'rb') as f:
            caseta_counter = pkl.load(f)
        caseta_counter += 1

        with open('../pkl/caseta_counter.pkl', 'wb') as f:
            pkl.dump(caseta_counter, f)


    # print('Caseta {} enter service b station at time: {}'.format(str(name), str(env.now)))

    env.process(service_a(env, server_a, server_b, mu_a, mu_b, arrival_time, name, sigma_a, sigma_b))


def service_a(env, server_a, server_b, mu_a, mu_b, arrival_time, name, sigma_a, sigma_b):
    print('wait here')
    print(len(server_a.users))
    print('&&&&&&&&&&')
    print(len(server_b.users))
    with server_a.request() as req:
        yield req

        # service time
        ser_time = np.max(np.random.normal(mu_a, sigma_a, 1),0)


        yield env.timeout(ser_time)


    env.process(service_b(env, server_a, server_b, mu_a, mu_b, arrival_time, name, sigma_a, sigma_b))

def customer_arrivals(env, server_a, server_b,  mu_a, mu_b, total_caseta, end_time, sigma_a, sigma_b):

    name = 0
    caseta_counter = 0
    with open('../pkl/caseta_counter.pkl', 'wb') as f:
        pkl.dump(caseta_counter, f)

    while True:

        # get the external stream identity
        if name < total_caseta:

            yield env.timeout(0.001)
        else:
            yield env.timeout(end_time)

        arrival_time = env.now

        # update current customer

        name += 1
        env.process(service_a(env, server_a, server_b, mu_a, mu_b, arrival_time, name, sigma_a, sigma_b))


def main(args):

    throughput = []
    for num_caseta in tqdm(range(25, 40)):

        env = simpy.Environment()

        # define server stage a
        server_a_list = []
        for ind in range(args.num_servers_a):
            server_a_list.append(simpy.Resource(env, capacity=1))

        server_b_list = []
        for ind in range(args.num_servers_b):
            server_b_list.append(simpy.Resource(env, capacity=1))


        env.process(customer_arrivals(env, server_a_list, server_b_list,   args.mu_a, args.mu_b, num_caseta,  args.end_time,
                                      args.sigma_a, args.sigma_b))
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
    parser.add_argument('--end_time', type=float, help='end time in the sim', default=5000)
    parser.add_argument('--mu_a', type=float, help='service rate', default=1.49)
    parser.add_argument('--mu_b', type=float, help='service rate', default=2.52)
    parser.add_argument('--sigma_a', type=float, help='service rate', default=0.4)
    parser.add_argument('--sigma_b', type=float, help='service rate', default=0.4)
    parser.add_argument('--total_caseta', type=float, help='total number of casetas', default=5)
    parser.add_argument('--num_servers_a', type=float, help='total number of casetas', default=5)
    parser.add_argument('--num_servers_b', type=float, help='total number of casetas', default=11)
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
