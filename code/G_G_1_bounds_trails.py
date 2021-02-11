# imports
import simpy
import numpy as np
import sys
import argparse
import pandas as pd
import pickle as pkl
import os

def service(env, name, server, mu, arrival_time):

    with server.request() as req:
        yield req

        # service time
        ser_time = np.random.gamma(99, 0.10)

        if name > 1:
            with open('../pkl/avg_waiting.pkl', 'rb') as f:
                avg_waiting = pkl.load(f)
        else:
            avg_waiting = 0
            with open('../pkl/avg_waiting.pkl', 'wb') as f:
                pkl.dump(avg_waiting, f)

        waiting_time = env.now - arrival_time
        avg_waiting = (avg_waiting * name + waiting_time) / (name + 1)

        with open('../pkl/avg_waiting.pkl', 'wb') as f:
            pkl.dump(avg_waiting, f)

        yield env.timeout(ser_time)


        if env.now > 40000:
            print(avg_waiting)

def customer_arrivals(env, server, r, mu):

    name = 0
    while True:

        # get the external stream identity

        yield env.timeout(np.random.gamma(100, 0.10))

        arrival_time = env.now

        # update current customer

        name += 1
        env.process(service(env, name, server, mu, arrival_time))

    print('finish')

def main(args):


    env = simpy.Environment()

    server = simpy.Resource(env, capacity=1)
    env.process(customer_arrivals(env, server, args.r, args.mu))
    env.run(until=(args.end_time))


def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--end_time', type=float, help='end time in the sim', default=250000.)
    parser.add_argument('--r', type=float, help='arrival rate', default=0.75)
    parser.add_argument('--mu', type=float, help='service rate', default=1.)



    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)