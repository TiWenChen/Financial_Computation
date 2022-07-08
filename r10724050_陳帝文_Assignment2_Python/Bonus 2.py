import pandas as pd
import numpy as np
import scipy.stats
import random
import math

S0=50
K=50
r=0.1
q=0.05
sigma=0.4
T=0.5
n=10000
call='c'

def combinatorial(S0,K,r,q,sigma,T,n,call):
    if call=='c':
        call=1
    elif call=='p':
        call=-1
    
    u=np.exp(sigma*np.sqrt(T/n))
    d=1/u
    P=(np.exp((r-q)*T/n)-d)/(u-d)
    
    連加=0.0
    for i in range(n+1): #0~n
        bino=scipy.stats.binom.pmf(i, n, 1-P)
        S_t=S0*(u**(n-i))*(d**i)
        連加+=bino*max(call*(S_t-K),0)
    value=np.exp(-r*T)*連加
    return value
    
#combinatorial(S0,K,r,q,sigma,T,n,'c')
print("The value of the option by combinatorial method: %.4f" %combinatorial(S0,K,r,q,sigma,T,n,'c'))
print("The value of the option by combinatorial method: %.4f" %combinatorial(S0,K,r,q,sigma,T,n,'p'))