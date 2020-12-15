import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from sympy import *
import os
import pickle as pkl
import cmath
from scipy.integrate import quad



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
def A(s,lam1,lam2,mu2):
    return(lam1/(lam1+lam2-lam2*(ser_lap(s,mu2))))
def beta(s,lam1,lam2,mu1,mu2):
    return (lam1/(lam1+lam2+s)+((A(s,lam1,lam2,mu2)*lam2)/(lam1+lam2+s))*(ser_lap(s,mu2)-busy(s+lam1,lam2,mu2)))/(1-((lam2*busy(s+lam1,lam2,mu2))/(lam1+lam2+s)))
def tau(s,lam1,lam2,mu1,mu2):
    return ser_lap(s,mu1)*(A(s,lam1,lam2,mu2)*(1-F(lam1+lam2-lam2*busy(s+lam1,lam2,mu2),lam1,lam2,mu1,mu2))+F(lam1+lam2-lam2*busy(s+lam1,lam2,mu2),lam1,lam2,mu1,mu2)*beta(s,lam1,lam2,mu1,mu2))
def get_var(lam1, lam2, mu1, mu2):
    s = Symbol('s')
    y = tau(s,lam1,lam2,mu1,mu2)
    dx = diff(y, s)
    dxdx = diff(dx,s)
    return dxdx.subs(s, 0)-(dx.subs(s,0))**2
def choose_r(r):
    print('For r={}, the error plot is:' .format(r))
    n = np.arange(1, 50)
    plt.figure()
    plt.plot(n,r**(2*n))
    plt.show()
    return r**(2*n)

def factorial(n):
    return np.math.factorial(n)
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

def f_t_Fourier_(t,n, lam1,lam2,mu1,mu2,h = 1, b = 0.1):
    a = h*np.exp(b*t)/np.pi
    b = lap(b)
    c = 2*h*np.exp(b*t)/pi
    k = np.arange(1,n)
    d = np.sum((tau(b+1j*k*h,lam1,lam2,mu1,mu2).real)*np.cos(k*h*t))
    return N(a*b+c*d)


def f_t_Fourier_t(t,h = 0.02, b = 0.001):
    lam2 = 1
    lam1 = 1.8
    mu2 = 10
    mu1 = 2.5
    n2 = 5000
    a = h*np.exp(b*t)/np.pi
    b = lap(b)
    c = 2*h*np.exp(b*t)/pi
    k = np.arange(1,n2)
    d = np.sum((tau(b+1j*k*h,lam1,lam2,mu1,mu2).real)*np.cos(k*h*t))
    if N(a*b+c*d)<0:
        return 0
    else:
        return (N(a*b+c*d)+0.012)

def main():

    lam2 = 1
    lam1 = 1.8
    mu2 = 10
    mu1 = 2.5
    n2 = 5000
    print(f_t_Fourier_(1, n2, lam1, lam2, mu1, mu2, 0.02, 0.001))
    tt = Symbol('tt')
    res, err = quad(f_t_Fourier_t ,  0, 5)
    print(res, err)

if __name__ == '__main__':
    main()
