from utils_ph import *
import sympy
from sympy import *
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
import pickle as pkl
import time
import numpy as np

def analytical_expression(result,  mu_0, lam_0, lam_1, mu_1,  vals):
    x, y, z, t = symbols('x y z t')

    for ind in tqdm(range(result.shape[0])):
        num_mu0 = int(result.loc[ind, 'mu0'])
        numlam0lam1 = int(result.loc[ind, 'lam0lam1'])
        nummu0lam0lam1 = int(result.loc[ind, 'lam0lam1mu0'])
        if ind == 0:
            expression = simplify(give_tail_analytic(num_mu0, numlam0lam1, nummu0lam0lam1, mu_0, lam_0, lam_1, mu_1,x)) * result.loc[ind, 'prob']
        else:
            expression += simplify(give_tail_analytic(num_mu0, numlam0lam1, nummu0lam0lam1, mu_0, lam_0, lam_1, mu_1,x)) * result.loc[ind, 'prob']

        expression = simplify(expression)

    start_time = time.time()
    bb = expression.subs(x, vals[0])
    end_time = time.time()

    time1 = end_time-start_time

    start_time = time.time()
    bb = expression.subs(x, vals[1])
    end_time = time.time()

    time2 = end_time - start_time

    return np.mean(np.array([time1, time2]))

    # pkl.dump(expression, open(analytic_exp_path, 'wb'))

