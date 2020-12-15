import argparse
import sys
# import numpy as np
# from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import get_mean_queue, get_marginal_probs_sum, get_limit_prob_single, get_limit_prob
from utils import get_limit_prob_center, get_multi_cofficienet, service_rate_state_dependent, eval_lamnda
from utils import get_lin_vals, get_lin_param, init_p_matrix

def main(args):

    assert args.r.shape == (args.number_of_centers, args.number_of_classes), 'Wrong dimensions of r'

    args.p = init_p_matrix(args)  # insert transition matrix values

    lamb_mat = eval_lamnda(args)  # get effective lambda values

    # get values for marginal dist:
    if args.marginal:
        for center_ind in range(args.number_of_centers):
            num_servers = np.zeros(args.number_of_centers)
            for num_servers_in_center in range(args.min_servers[center_ind], args.max_servers[center_ind]):
                num_servers[center_ind] = num_servers_in_center

                limitprob = get_limit_prob_single(args, lamb_mat, num_servers, center_ind, True)

                limitprob = limitprob / np.sum(limitprob)

                mean_queue = np.sum(np.arange(limitprob.shape[0]) * limitprob)

                print(center_ind, num_servers_in_center, mean_queue)



    if args.total_network:
        args.max_servers += 1 # because iterator does not include the upper limit
        x = np.zeros((args.max_servers[0]-args.min_servers[0], args.max_servers[1]-args.min_servers[1]))
        y = np.copy(x)
        total_cost = np.copy(x)
        waiting_cost = []
        staffing_cost = []
        allocation = []
        for num_in_center_1 in tqdm(range(args.min_servers[0], args.max_servers[0])):
            for num_in_center_2 in range(args.min_servers[1], args.max_servers[1]):

                num_servers = np.array([num_in_center_1, num_in_center_2])

                limitprob, limitprob_ = get_limit_prob(args, lamb_mat, num_servers, True, args.n_max)

                limitprob = limitprob/np.sum(limitprob)

                limitprob_ = limitprob_ / np.sum(limitprob_)

                margin_0 = np.sum(limitprob_, axis=1)
                margin_1 = np.sum(limitprob_, axis=0)

                # marginal_prob_sum = get_marginal_probs_sum(limitprob_, args.n_max)

                # get x and y values of the 3D plot
                x[num_in_center_1 - args.min_servers[0], num_in_center_2 - args.min_servers[1]] = num_in_center_1
                y[num_in_center_1 - args.min_servers[0], num_in_center_2 - args.min_servers[1]] = num_in_center_2

                total_cost[num_in_center_1 - args.min_servers[0], num_in_center_2 - args.min_servers[1]] = \
                    args.C_w[0]*get_mean_queue(margin_0) + args.C_w[1]*get_mean_queue(margin_1) \
                             + args.C_s[0]*num_servers[0] + args.C_s[1]*num_servers[1]


                allocation.append((num_in_center_1, num_in_center_2))


                # print(num_in_center_1, num_in_center_2, mean_queue)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(x, y, total_cost, cmap='viridis', edgecolor='none')
    ax.set_title('Cost per number of ser vers')
    plt.xlabel('number of servers center 0')
    plt.ylabel('number of servers center 1')

    plt.show()
    print(np.sum(limitprob))
    print(limitprob[0])



def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=np.array, help='external arrivals', default=np.array([[2, 0.25],[0.25, 3]]))
    parser.add_argument('--p', type=np.array, help='transision matrix', default=np.array([]))
    parser.add_argument('--number_of_centers', type=int, help='number of centers', default=2)
    parser.add_argument('--number_of_classes', type=int, help='number of classes', default=2)
    parser.add_argument('--mu', type=np.array, help='service rates', default=np.array([3, 3.8]))
    parser.add_argument('--total_network', type=bool, help='eval steady of total net?', default=True)
    parser.add_argument('--marginal', type=bool, help='eval steady of marignal net?', default=False)
    parser.add_argument('--C_w', type=np.array, help='waiting cost', default=np.array([5, 5]))
    parser.add_argument('--C_s', type=np.array, help='Staffing cost', default=np.array([1, 1]))
    parser.add_argument('--min_servers', type=np.array, help='min possible number of servers', default=np.array([1, 1]))
    parser.add_argument('--max_servers', type=np.array, help='max possible number of servers', default=np.array([3, 3]))
    parser.add_argument('--n_max', type=int, help='numerical_limit for steady-state', default=100)

    args = parser.parse_args(argv)

    return args



if __name__ =='__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)