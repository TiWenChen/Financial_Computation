import pandas as pd
import numpy as np
import scipy.stats
import random
import math

def BS(S0,K,r,q,sigma,T,call):
    if call=='c':
        call=1
    elif call=='p':
        call=-1
    d1=((np.log(S0/K)+(r-q+0.5*sigma**2)*T)/(sigma*math.sqrt(T)))
    d2=(d1-sigma*math.sqrt(T))
    value=(S0*np.exp(-q*T)*scipy.stats.norm.cdf(call*d1)-K*np.exp(-r*T)*scipy.stats.norm.cdf(call*d2))*call
    return value

def Monte_Carlo(S0,K,r,q,sigma,T,repetition_times,samples,call):
    if call=='c':
        call=1
    elif call=='p':
        call=-1
    payoff=[]
    for i in range(repetition_times):
        ST=np.exp(scipy.stats.norm.rvs(size=samples,loc=np.log(S0)+(r-q-0.5*(sigma**2))*T,scale=(sigma)*math.sqrt(T)))
        simulation_payoff=np.exp(-r*T)*(np.maximum(call*(ST-K),0))
        payoff.append(np.mean(simulation_payoff))
    upper_bond=np.mean(payoff)+2*np.std(payoff)
    lower_bond=np.mean(payoff)-2*np.std(payoff)
    print("The 95 percent of the Monte Carlo interval %.4f ~ %.4f"%(lower_bond, upper_bond))
    #return lower_bond, upper_bond

def CRR_Binomial(S0,K,r,q,sigma,T,n,call,otype):
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
    
    #underlying assets price matrix
    S=np.zeros((n+1,n+1))
    S[0,0]=S0
    for i in range(1,n+1):
        for j in range(i+1):
            S[j,i]=S[0,0]*(u**(i-j))*(d**(j))
    
    #interval value
    iv=np.zeros((n+1,n+1))
    for i in range(0,n+1):
        for j in range(i+1):
            iv[j,i]=max(call*(S[j,i]-K),0)
    
    #options price
    C=np.zeros((n+1,n+1))
    C[...,n]=iv[...,n]#np.maximum(call*(S[...,n]-K),0)
    for i in range(n-1,-1,-1):
        for j in range(i+1):
            C[j,i]=max(np.exp(-r*T/n)*(P*C[j,i+1]+(1-P)*C[j+1,i+1]),otype*iv[j,i])
    #print(C)
    return float(C[0,0])

S0=50
K=50
r=0.1
q=0.05
sigma=0.4
T=0.5
n=100
repetition_times=20
samples=10000
call='c'
otype='E'

print("The value of the call option calculated by Black Scholes model: %.4f"%BS(S0, K, r, q, sigma, T, 'c'))
print("The value of the put option calculated by Black Scholes model: %.4f"%BS(S0, K, r, q, sigma, T, 'p'))
Monte_Carlo(S0, K, r, q, sigma, T, repetition_times, samples, 'c')
Monte_Carlo(S0, K, r, q, sigma, T, repetition_times, samples, 'p')

data=pd.DataFrame()


for i in [100, 500, 1000, 2000, 3000]:

    E_c=CRR_Binomial(S0, K, r, q, sigma, T, i, 'c', 'E')
    A_c=CRR_Binomial(S0, K, r, q, sigma, T, i, 'c', 'A')
    E_p=CRR_Binomial(S0, K, r, q, sigma, T, i, 'p', 'E')
    A_p=CRR_Binomial(S0, K, r, q, sigma, T, i, 'p', 'A')
    data['n= %d' %i] = [E_c, A_c, E_p, A_p]

data.index=['European call', 'American call', 'European put', 'American put']
print(data)

