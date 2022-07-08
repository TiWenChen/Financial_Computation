import pandas as pd
import numpy as np
import scipy.stats
import math

def CRR_one_vector(S0,K,r,q,sigma,T,n,call,otype):
    if call=='c':
        call=1
    elif call=='p':
        call=-1
    
    if otype=='E':
        otype=0
    elif otype=='A':
        otype=1
    
    u=np.exp(sigma*math.sqrt(T/n))
    d=1/u
    P=(np.exp((r-q)*T/n)-d)/(u-d)
    
    c=np.zeros(n+1)
    c[0]=S0
    for i in range(1,n+1): #i 是橫向
        for j in range(i,-1,-1): #j 是縱向
            c[j]= S0*(u**(i-j))*(d**j)
            #print(i,j,i-j-j,c[0],'//',u**(i-j)*d**j)
    c=np.maximum(call*(c-K),0)
    #print(c)
    for i in range(n, -1, -1):
        for j in range(i):
            #c[j]=round(max(np.exp(-r*T/n)*(c[j]*P+c[j+1]*(1-P)),otype*(S0*(u**(i-1-j))*(d**(j))-K)),4)
            c[j]=round(max(np.exp(-r*T/n)*(c[j]*P+c[j+1]*(1-P)),otype*(call*(S0*(u**(i-1-j))*(d**(j))-K))),6)
    return (c[0])

S0=50
K=50
r=0.1
q=0.05
sigma=0.4
T=0.5
n=500
repetition_times=20
samples=10000
call='c'
otype='E'
"""
print("The value of the option by binomial tree method: %.4f"%CRR_one_vector(S0,K,r,q,sigma,T,n,'c','E'))
print("The value of the option by binomial tree method: %.4f"%CRR_one_vector(S0,K,r,q,sigma,T,n,'c','A'))
print("The value of the option by binomial tree method: %.4f"%CRR_one_vector(S0,K,r,q,sigma,T,n,'p','E'))
print("The value of the option by binomial tree method: %.4f"%CRR_one_vector(S0,K,r,q,sigma,T,n,'p','A'))
"""
ary=[100, 500, 1000, 2000, 3000]

for i in ary:
    print("Option value: ", CRR_one_vector(S0, K, r, q, sigma, T, i, call, otype))
