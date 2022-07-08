import pandas as pd
import numpy as np
import scipy.stats


def Finite_Difference_Method(S0, K, r, q, sigma, T, S_min, S_max, m, n, call, otype, mtype):
    if call=='c':
        call=1
        f_m=S_max-K
        f_0=0
    elif call=='p':
        call=-1
        f_m=0
        f_0=K
    if otype=='E':
        otype=0
    elif otype=='A':
        otype=1
    
    delta_S=(S_max-S_min)/m
    delta_t=(T-0)/n

    #S0 在第幾個j
    
    count=int((S0/delta_S))
    
    
    if mtype=='I':
        #construct A
        A=np.zeros((m-1, m+1))
        for j, l in enumerate(A):#j 是縱向
            J=m-1-j ###################################################
            a_j=0.5*(r-q)*J*delta_t-0.5*(sigma**2)*(J**2)*delta_t
            b_j=1+(sigma**2)*(J**2)*delta_t+r*delta_t
            c_j=(-0.5*(r-q))*J*delta_t-0.5*(sigma**2)*(J**2)*delta_t
            A[j, j]=c_j
            A[j, j+1]=b_j
            A[j, j+2]=a_j
        c_m_1=A[0,0]
        a_1=A[-1,-1]
        #最後一期 f_value
        option_v_next=np.maximum(call*(np.linspace(S_max, S_min, m+1)-K),0)
        option_v_next=option_v_next[1:-1] 
        A=A[...,1:-1]
        S=np.linspace(S_max, S_min, m+1) #stock price
        S=S[1:-1]

        for i in (range(1, n+1)): 
            option_v_next[0]=option_v_next[0]-c_m_1*f_m
            option_v_next[-1]=option_v_next[-1]-a_1*f_0
            option_v=np.dot(np.linalg.inv(A),option_v_next)
            option_v[option_v<=0]=0
            option_v=np.maximum(option_v, otype*np.maximum(call*(S-K), 0))
            option_v_next=option_v
            
        option_v=np.insert(option_v, 0, f_m)
        option_v=np.insert(option_v, len(option_v), f_0)
        
    if mtype=='E':
        option_v=np.maximum(call*(np.linspace(S_max, S_min, m+1)-K),0)
        #construct A
        A=np.zeros((m-1, m+1))
        for j, l in enumerate(A):#j 是縱向
            J=m-1-j ##################################################
            a_j=(1/(1+r*delta_t))*(-0.5*(r-q)*J*delta_t+0.5*(sigma**2)*(J**2)*delta_t)
            b_j=(1/(1+r*delta_t))*(1-(sigma**2)*(J**2)*delta_t)
            c_j=(1/(1+r*delta_t))*(0.5*(r-q)*J*delta_t+0.5*(sigma**2)*(J**2)*delta_t)
            A[j, j]=c_j
            A[j, j+1]=b_j
            A[j, j+2]=a_j
        S=np.linspace(S_max, S_min, m+1) #stock price
        
        for i in (range(1,n+1)):
            option_v=np.dot(A, option_v)
            option_v=np.insert(option_v, 0, f_m)
            option_v=np.insert(option_v, len(option_v), f_0)
            option_v[option_v<0]=0
            option_v=np.maximum(option_v,otype*np.maximum(call*(S-K), 0))
            
    return (option_v[m-count])

S0=50
K=50
r=0.05
q=0.01
sigma=0.4
T=0.5
S_max=100
S_min=0

print("Finite difference: ", round(Finite_Difference_Method(S0, K, r, q, sigma, T, S_min, S_max, 400, 100, 'c', 'E', 'I'), 4))
print("Finite difference: ", round(Finite_Difference_Method(S0, K, r, q, sigma, T, S_min, S_max, 100, 1000, 'c', 'E', 'E'),4))
print("Finite difference: ", round(Finite_Difference_Method(S0, K, r, q, sigma, T, S_min, S_max, 400, 100, 'p', 'E', 'I'),4))
print("Finite difference: ", round(Finite_Difference_Method(S0, K, r, q, sigma, T, S_min, S_max, 100, 1000, 'p', 'E', 'E'),4))

print("Finite difference: ", round(Finite_Difference_Method(S0, K, r, q, sigma, T, S_min, S_max, 400, 100, 'c', 'A', 'I'),4))
print("Finite difference: ", round(Finite_Difference_Method(S0, K, r, q, sigma, T, S_min, S_max, 100, 1000, 'c', 'A', 'E'),4))
print("Finite difference: ", round(Finite_Difference_Method(S0, K, r, q, sigma, T, S_min, S_max, 400, 100, 'p', 'A', 'I'),4))
print("Finite difference: ", round(Finite_Difference_Method(S0, K, r, q, sigma, T, S_min, S_max, 100, 1000, 'p', 'A', 'E'),4))