import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import *
import sys
sys.path.append(r'G:\My Drive\butools2\Python')
sys.path.append('/home/d/dkrass/eliransc/Python')
from tqdm import tqdm
from butools.ph import *
from butools.map import *
from butools.queues import *
import time
from tqdm import tqdm
from butools.mam import *
from butools.dph import *
from scipy.linalg import expm, sinm, cosm
import pickle as pkl
from numpy.linalg import matrix_power


def busy(s, lam2, mu2):
    return ((lam2 + mu2 + s) - ((lam2 + mu2 + s) ** 2 - 4 * lam2 * mu2) ** 0.5) / (2 * lam2)


def ser_lap(s, mu):
    return mu / (s + mu)


def hyper(s, lam1, lam2, mu1, mu2):
    return ser_lap(s, mu1) * lam1 / (lam1 + lam2) + ser_lap(s, mu2) * lam2 / (lam1 + lam2)


def rho(lam1, lam2, mu1, mu2):
    return (lam1 + lam2) * ((lam1 / ((lam1 + lam2) * mu1)) + (lam2 / ((lam1 + lam2) * mu2)))


def w_lap(s, lam1, lam2, mu1, mu2):
    return ((1 - rho(lam1, lam2, mu1, mu2)) * s) / (s - (lam1 + lam2) * (1 - hyper(s, lam1, lam2, mu1, mu2)))


def F(s, lam1, lam2, mu1, mu2):
    return w_lap(s, lam1, lam2, mu1, mu2) * ser_lap(s, mu1)


def A(s, lam1, lam2, mu2):
    return (lam1 / (lam1 + lam2 - lam2 * (ser_lap(s, mu2))))


def beta(s, lam1, lam2, mu1, mu2):
    return (lam1 / (lam1 + lam2 + s) + ((A(s, lam1, lam2, mu2) * lam2) / (lam1 + lam2 + s)) * (
                ser_lap(s, mu2) - busy(s + lam1, lam2, mu2))) / (
                       1 - ((lam2 * busy(s + lam1, lam2, mu2)) / (lam1 + lam2 + s)))


def tau(s, lam1, lam2, mu1, mu2):
    return ser_lap(s, mu1) * (A(s, lam1, lam2, mu2) * (
                1 - F(lam1 + lam2 - lam2 * busy(s + lam1, lam2, mu2), lam1, lam2, mu1, mu2)) + F(
        lam1 + lam2 - lam2 * busy(s + lam1, lam2, mu2), lam1, lam2, mu1, mu2) * beta(s, lam1, lam2, mu1, mu2))


def get_var(lam1, lam2, mu1, mu2):
    s = Symbol('s')
    y = tau(s, lam1, lam2, mu1, mu2)
    dx = diff(y, s)
    dxdx = diff(dx, s)
    return dxdx.subs(s, 0) - (dx.subs(s, 0)) ** 2


def get_n_moment(lam0, lam1, mu0, mu1, n):
    s = Symbol('s')
    y = tau(s, lam0, lam1, mu0, mu1)
    moments = []
    for i in tqdm(range(1, n + 1)):
        if i == 1:
            dx = diff(y, s)
        else:
            dx = diff(dx, s)

        moments.append(dx.subs(s, 0) * ((-1) ** (i)))

    return moments


def get_C_first(T01, T02, s):
    krom_sum = kroneker_sum(T01, T02)

    return krom_sum


def get_C(T01, T02, alpha1, alpha2, s):
    krom_sum = kroneker_sum(np.dot(T01, alpha1), np.dot(T02, alpha2))
    krom_sum = kroneker_sum(krom_sum, np.zeros(s.shape))
    return krom_sum


def get_C_0(T01, T02, alpha1, alpha2, Beta):
    krom_sum = kroneker_sum(np.dot(T01, alpha1), np.dot(T02, alpha2))
    krom_sum = np.kron(krom_sum, Beta)

    return krom_sum


def get_B(T1, T2, s):
    krom_sum = kroneker_sum(T1, T2)

    return kroneker_sum(krom_sum, s)


def get_A(T1, T2, Beta, s0):
    kron_sum = kroneker_sum(np.zeros(T1.shape), np.zeros(T2.shape))
    kron_sum = kroneker_sum(kron_sum, np.dot(s0, Beta))
    return kron_sum


def get_A0(T1, T2):
    krom_sum = kroneker_sum(T1, T2)
    return krom_sum


def kroneker_sum(G, H):
    size_g = G.shape[0]
    size_h = H.shape[0]
    return np.kron(G, np.identity(size_h)) + np.kron(np.identity(size_g), H)


def get_ro(lam1, lam_ext, mu11):
    return (lam1 + lam_ext) / mu11

def get_steady_ph_sys(lam_1, lam_ext, mu_11, path_ph, ub_v):

    alpha1, T1 = pkl.load(open(path_ph, 'rb'))
    T01 = -np.dot(T1, np.ones((T1.shape[0], 1)))
    print('T1 is loaded')
    alpha2 = np.array([1])
    T2 = np.array([-lam_ext]).reshape((1, 1))
    T02 = -np.dot(T2, np.ones((T2.shape[0], 1)))

    s = np.array([[-mu_11]]).reshape((1, 1))
    Beta = np.array([1.]).reshape((1, 1))
    s0 = -np.dot(s, np.ones((s.shape[0], 1)))

    A0 = get_A0(T1, T2)
    A1 = np.kron(np.identity(A0.shape[0]), s0)
    AA = get_A(T1, T2, Beta, s0)
    B = get_B(T1, T2, s)
    C = get_C(T01, T02, alpha1, alpha2, s)
    C0 = get_C_0(T01, T02, alpha1, alpha2, Beta)

    from scipy.spatial import distance_matrix

    epsilon = 10 ** (-40)
    R = np.zeros(AA.shape)
    from scipy.spatial import distance
    for i in tqdm(range(700)):
        R_curr = -np.dot((np.dot(matrix_power(R, 2), AA) + C), np.linalg.inv(B))
        dst = np.sum(np.square(R_curr - R))
        R = R_curr
        # print(dst)
        if dst < epsilon:
            break
    print(dst)

    # pkl.dump(R, open('../pkl/R_'+str(ub_v)+'.pkl', 'wb'))

    rho_value = get_ro(lam_1, lam_ext, mu_11)
    n = R.shape[0]
    a = np.zeros((2 * n, 2 * n))

    ## computing a

    # A0
    for i in range(n):
        for j in range(n):
            a[i, j] = A0[j, i]
    # A1
    for i in range(n):
        for j in range(n, 2 * n):
            a[i, j] = A1[j - n, i]

    # C0
    for i in range(n - 1):
        for j in range(n):
            a[i + n, j] = C0[j, i]

    CC = B + np.dot(R, AA)

    # B+R.A
    for i in range(n - 1):
        for j in range(n, 2 * n):
            a[i + n, j] = CC[j - n, i]

    # p0*e = 1-rho
    a[2 * n - 1, :n] = 1

    ## computing b

    b = np.zeros((2 * n, 1))
    b[-1, 0] = 1 - rho_value

    ## solving steady-state

    x = np.linalg.solve(a, b)

    ## exracting u1

    u1 = x[n:].reshape((1, n))

    ## computing average number of customers

    avg_number = np.sum(np.dot(u1, matrix_power(np.identity(R.shape[0]) - R, -2)))


    ## Rh0 of mm1

    rho_mm1 = (lam_ext + lam_1) / mu_11

    ## avg_mm1

    avg_mm1 = rho_mm1/(1-rho_mm1)

    print('The mm1 avg is {} and the true avg is: {} '.format(avg_mm1, avg_number))
    pkl.dump((R, x), open('../pkl/R_' + str(ub_v) + '.pkl', 'wb'))

    return avg_number

if __name__ =='__main__':

    lam_0 = 0.1
    lam_1 = 1 - lam_0
    mu_0 = 0.7
    mu_1 = 3.
    mu_11 = 1.5
    lam_ext = 0.5

    get_steady_ph_sys(lam_1, lam_ext, mu_11, 'path_ph')