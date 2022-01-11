import numpy as np
import os
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.special import comb
from sympy import *
import cmath
from scipy.linalg import expm, sinm, cosm
from tqdm import tqdm
from scipy.integrate import quad
from scipy.stats import erlang
import argparse
import sys
sys.path.append(r'G:\My Drive\butools2\Python')
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/butools/Python')

from butools.ph import *
from butools.map import *
from butools.queues import *

def busy(s,lam2,mu2):
    return ((lam2+mu2+s)-((lam2+mu2+s)**2-4*lam2*mu2)**0.5)/(2*lam2)
def ser_lap(s,mu):
    return mu/(s+mu)
def hyper(s,lam1,lam2,mu1,mu2):
    return ser_lap(s,mu1)*lam1/(lam1+lam2)+ser_lap(s,mu2)*lam2/(lam1+lam2)
def rho(lam1,lam2,mu1,mu2):
    return (lam1+lam2)*((lam1/((lam1+lam2)*mu1))+(lam2/((lam1+lam2)*mu2)))
def w_lap(s,lam1,lam2,mu1,mu2):
    return ((1-rho(lam1,lam2,mu1,mu2))*s)/(s-(lam1+lam2)*(1-hyper(s,lam1,lam2,mu1,mu2)))
def F(s,lam1,lam2,mu1,mu2):
    return w_lap(s,lam1,lam2,mu1,mu2)*ser_lap(s,mu1)
def A_(s,lam1,lam2,mu2):
    return(lam1/(lam1+lam2-lam2*(ser_lap(s,mu2))))
def beta(s,lam1,lam2,mu1,mu2):
    return (lam1/(lam1+lam2+s)+((A_(s,lam1,lam2,mu2)*lam2)/(lam1+lam2+s))*(ser_lap(s,mu2)-busy(s+lam1,lam2,mu2)))/(1-((lam2*busy(s+lam1,lam2,mu2))/(lam1+lam2+s)))
def tau(s,lam1,lam2,mu1,mu2):
    return ser_lap(s,mu1)*(A_(s,lam1,lam2,mu2)*(1-F(lam1+lam2-lam2*busy(s+lam1,lam2,mu2),lam1,lam2,mu1,mu2))+F(lam1+lam2-lam2*busy(s+lam1,lam2,mu2),lam1,lam2,mu1,mu2)*beta(s,lam1,lam2,mu1,mu2))
def get_var(lam1, lam2, mu1, mu2):
    s = Symbol('s')
    y = tau(s,lam1,lam2,mu1,mu2)
    dx = diff(y, s)
    dxdx = diff(dx,s)
    return dxdx.subs(s, 0)-(dx.subs(s,0))**2

def ln_t(t,n):
    return np.exp(-t/2)*Ln_t(t,n)
def Ln_t(t,n):
    arr = np.arange(n+1)
    return np.sum(comb(n, arr)*((-t)**arr)/np.array([factorial(i) for i in arr]))
def lap(s,mu = 1):
    return mu/(mu+s)
def Q(z):
    return lap((1+z)/(2*(1-z)))/(1-z)
def q_n(n, r = 0.5):
    x = np.linspace(1,2*n, 2*n)
    a = 1/(2*n*(r**n))
    b = np.sum(((-1)**x)*(Q(r*np.exp(r*np.exp(1j*np.pi*x/n))).real))
    return a*b
def f_t(t,n):
    n_arr = np.arange(1,n)
    res_ = np.array([])
    for n in n_arr:
        res_ = np.append(res_,ln_t(t,n)*q_n(n, 0.5))
    return np.sum(res_)
def factorial(n):
    return np.math.factorial(n)

from scipy.special import factorial
def p_v(x, v,lam):
    return np.exp(-lam*x)*((lam*x)**v)/factorial(v)

def prob_v_integ(x,v,lam, A, a):
    s0 = -np.dot(A, np.ones((A.shape[0],1)))
    return np.dot(np.dot(a,expm(A*x)),s0)*np.exp(-x*lam)*((lam*x)**v)/factorial(v)
def p_n(n,lam, A,a,  UL = 30):
    res = np.array([])
    for v in range(n,UL):
        res = np.append(res, v_prob(v,lam, A, a)[0]/(v+1))
    return np.sum(res)

def v_prob(v,lam, A, a, UL = 30):
    return quad(prob_v_integ,0,UL,args=(v, lam, A, a))

def dens_ph(x, A, a):
    s0 = -np.dot(A, np.ones((A.shape[0],1)))
    return np.dot(np.dot(a,expm(A*x)),s0)

def tail_ph(x, A, a):
    return np.dot(np.dot(a,expm(A*x)),np.ones((A.shape[0],1)))

def f_z_smaller_x_(x,aa,n, A, a):#
    return dens_ph(x, A,a)/erlang.cdf(x, n, loc=0, scale=1)
def f_z_smaller_x(aa, n, UL = 30):
    return quad(f_z_smaller_x_,aa,UL,args=(aa,n))[0]

def f_z_smaller__x(aa,n,A, a, UL =30):
    return erlang.pdf(aa, n, loc=0, scale=1)*quad(f_z_smaller_x_,aa,UL,args=(aa,n, A, a))[0]

def res_given_a_r(r,aa,A, a):
    val_mat = (tail_ph(aa, A, a)-tail_ph(r+aa, A, a))/tail_ph(aa, A, a)
    return val_mat[0,0]

def R_given_A_(aa,r,n, A ,a , A_curr, a_curr):
    return res_given_a_r(r, aa, A, a)*dens_ph(aa, A_curr, a_curr)

def f_z_smaller__x_moments(aa,n, moment, A, a):
    return (aa**moment)*f_z_smaller__x(aa,n, A,a)


def get_n_moment(lam1, lam2, mu1, mu2, n):
    s = Symbol('s')
    y = tau(s,lam1,lam2,mu1,mu2)
    for i in range(n):
        if i == 0:
            dx = diff(y,s)
        else:
            dx = diff(dx,s)
    return dx.subs(s, 0)

def main(args):

    ###################################################
    ## Get the PH of the non-Poisson inter-arrival
    ###################################################

    lam1 = 1.5
    lam2 = 1.
    mu1 = 2.
    mu2 = 6.0
    UL = 50
    ## check if the input is feasible
    rho__ = rho(lam1,lam2,mu1,mu2)
    print('Rho is: ', rho__)

    assert rho__ < 1, 'Not feasible input paramets lamdas and mus'

    pck_signature = 'pkl/'+str(lam1) + '_' + str(lam2) + '_' + str(mu1) + '_' + str(mu2) + '_UL_50.pkl'
    if os.path.exists(pck_signature):
        with open(pck_signature, 'rb') as f:
            a,A = pkl.load(f)
    else:
        moms = []
        for n in range(1, 6):
            moms.append(float(get_n_moment(lam1, lam2, mu1, mu2, n) * (-1) ** n))

        print(moms)
        try:
            a, A = PH3From5Moments(moms)
        except:
            a, A = PH2From3Moments(moms[:3])
        print(a, A)

        with open(pck_signature, 'wb') as f:
            pkl.dump((a,A), f)

    ###################################################
    ## Get the PH of the conditional erlang
    ###################################################
    pck_signature = 'pkl/'+str(lam1) + '_' + str(lam2) + '_' + str(mu1) + '_' + str(mu2) + '_age_given_n_UL_50.pkl'

    if os.path.exists(pck_signature):
        with open(pck_signature,'rb') as f:
            f_N_n_dict = pkl.load(f)
    else:

        f_N_n_dict = {}

        for n in tqdm(range(1, 10)):
            f_z_moments = []

            for moment in tqdm(range(1, 6)):
                f_z_moments.append(quad(f_z_smaller__x_moments, 0, UL, args=(n, moment, A, a))[0])

            try:
                a_curr, A_curr = PH3From5Moments(f_z_moments)
            except:
                print('PH level 2')
                print(f_z_moments)
                a_curr, A_curr = PH2From3Moments(f_z_moments[:3])

            f_N_n_dict[str(n) + '_a'] = a_curr
            f_N_n_dict[str(n) + '_A'] = A_curr


        with open(pck_signature, 'wb') as f:
            pkl.dump(f_N_n_dict, f)

    r_vals = np.linspace(0,7, 2500)
    P_R_r = []
    for r in tqdm(r_vals):
        start_time = time.time()

        p_R_given_n = np.array([])
        # for n = 0
        n = 0
        P_N_n = p_n(n, lam2, A, a, UL)
        R_given_A = 1 - tail_ph(r, A, a)
        # print(R_given_A[0, 0] * P_N_n)
        p_R_given_n = np.append(p_R_given_n, R_given_A * P_N_n)

        for n in range(1, 10):

            R_given_A = quad(R_given_A_, 0, UL, args=(r, n, A,a, f_N_n_dict[str(n) + '_A'], f_N_n_dict[str(n) + '_a']))[0]
            P_N_n = p_n(n,lam2,A, a,   UL)

            p_R_given_n = np.append(p_R_given_n, R_given_A*P_N_n)


        P_R_r.append(np.sum(p_R_given_n))

        # print("--- %s seconds ---" % (time.time() - start_time))

    pkl_signature = 'pkl/'+str(lam1)+'_'+str(lam2)+'_'+str(mu1)+'_'+str(mu2)+'P_R_r.pkl'
    with open(pkl_signature, 'wb') as f:
        pkl.dump((r_vals,P_R_r), f)


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