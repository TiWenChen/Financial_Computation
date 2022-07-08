import pandas as pd
import numpy as np
import scipy.stats 


def BS(S0,K,r,q,sigma,T,call):
    if call=='c':
        call=1
    elif call=='p':
        call=-1
    d1=((np.log(S0/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T)))
    d2=(d1-sigma*np.sqrt(T))
    value=(S0*np.exp(-q*T)*scipy.stats.norm.cdf(call*d1)-K*np.exp(-r*T)*scipy.stats.norm.cdf(call*d2))*call
    return value

def CRR_one_vector(S0,K,r,q,sigma,T,n,call,otype):
    if call=='c':
        call=1
    elif call=='p':
        call=-1
    
    if otype=='E':
        otype=0
    elif otype=='A':
        otype=1
    
    u=np.exp(sigma*np.sqrt(T/n))
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

def Bisection(pricing_model, S0, K, r, q, T, mprice, n, call, otype, sigma_upper_bond=1, sigma_lower_bond=0.0001, convergence=0.0001):
    
    if pricing_model=='BS':
        p1=BS(S0, K, r, q, sigma_upper_bond, T, call)
        p2=BS(S0, K, r, q, sigma_lower_bond, T, call)
        f1=p1-mprice
        f2=p2-mprice
       
        implied_volitility=float
        if f1*f2>0:
            print('p1: %.6f, p2: %.6f'%(p1,p2))
            print('The solutions of the equation is not odd. Please try again.')
            
        while ((f1*f2)<0 and (abs(sigma_upper_bond-sigma_lower_bond)>convergence)):
            x=0.5*(sigma_upper_bond+sigma_lower_bond)
            if (BS(S0, K, r, q, x, T, call)-mprice)*f1<0:
                sigma_lower_bond=x
                f2=BS(S0, K, r, q, sigma_lower_bond, T, call)-mprice
            elif (BS(S0, K, r, q, x, T, call)-mprice)*f2<0:
                sigma_upper_bond=x
                f1=BS(S0, K, r, q, sigma_upper_bond, T, call)-mprice
            elif x==0:
                implied_volitility=x
                break

            implied_volitility=0.5*(sigma_upper_bond+sigma_lower_bond)
        

    
    elif pricing_model=='CRR':
        p1=(CRR_one_vector(S0, K, r, q, sigma_upper_bond, T, n, call=call, otype=otype))
        p2=(CRR_one_vector(S0, K, r, q, sigma_lower_bond, T, n, call=call, otype=otype))
        implied_volitility=float
        #print(p1,p2)
        f1=p1-mprice
        f2=p2-mprice
        if f1*f2>0:
            print('The solutions of the equation is not odd. Please try again.')
        
        while ((f1*f2)<0 and (abs(sigma_upper_bond-sigma_lower_bond)>convergence)):
            
            x=0.5*(sigma_upper_bond+sigma_lower_bond)
            if (CRR_one_vector(S0, K, r, q, x, T, n, call, otype)-mprice)*f1<0:
                sigma_lower_bond=x
                f2=CRR_one_vector(S0, K, r, q, sigma_lower_bond, T, n, call, otype)-mprice
            elif (CRR_one_vector(S0, K, r, q, x, T, n, call, otype)-mprice)*f2<0:
                sigma_upper_bond=x
                f1=CRR_one_vector(S0, K, r, q, sigma_upper_bond,  T, n, call, otype)-mprice
            elif x==0:
                implied_volitility=x
                break
        
        implied_volitility=0.5*(sigma_upper_bond+sigma_lower_bond)

    return implied_volitility

def BS_differentiate(S0, K, r, q, sigma, T):
    d1=(np.log(S0/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    d2_d=-np.sqrt(T)-(np.log(S0/K)+(r-q-0.5*sigma**2)*T)/(np.sqrt(T)*sigma**2)
    d1_d=d2_d+np.sqrt(T)
    f_d=S0*np.exp(-q*T)*np.exp(-0.5*d1**2)*d1_d/np.sqrt(2*np.pi)-K*np.exp(-r*T)*np.exp(-0.5*d2**2)*d2_d/np.sqrt(2*np.pi)
    
    return f_d
    

def Newton(pricing_model, S0, K, r, q, T, mprice, n, call, otype, initial_sigma=0.8, convergence=0.0001):
    
    if pricing_model=='BS':
        p1=BS(S0, K, r, q, initial_sigma, T, call)
        f1=p1-mprice
        new_sigma=initial_sigma-f1/BS_differentiate(S0, K, r, q, initial_sigma, T)
        if abs(new_sigma)>1000:
            print('Please guess a new initial sigma.')
        while ((abs(new_sigma-initial_sigma)>0.0001) and abs(new_sigma)<1000):
            initial_sigma=new_sigma
            p1=BS(S0, K, r, q, initial_sigma, T, call)
            f1=p1-mprice
            new_sigma=initial_sigma-f1/BS_differentiate(S0, K, r, q, initial_sigma, T)
            
            if abs(new_sigma)>1000:
                print('Please guess a new initial sigma.')
                break
        implied_volatility=0.5*(new_sigma+initial_sigma)
    
    elif pricing_model=='CRR':
        p1=CRR_one_vector(S0,K,r,q,initial_sigma,T,n,call,otype)
        f1=p1-mprice
        sigma_m=(CRR_one_vector(S0,K,r,q,initial_sigma+0.1,T,n,call,otype)-p1)/0.1 #斜率
        
        new_sigma=initial_sigma-f1/sigma_m
        if abs(new_sigma)>1000:
            print('Please guess a new initial sigma.')
        
        while ((abs(new_sigma-initial_sigma)>0.0001) and abs(new_sigma)<1000):
            initial_sigma=new_sigma
            p1=CRR_one_vector(S0,K,r,q,initial_sigma,T,n,call,otype)
            f1=p1-mprice
            sigma_m=(CRR_one_vector(S0,K,r,q,initial_sigma+0.1,T,n,call,otype)-p1)/0.1 #斜率
            new_sigma=initial_sigma-f1/sigma_m
            #print(abs(new_sigma-initial_sigma))
            if abs(new_sigma)>1000:
                print('Please guess a new initial sigma.')
                break
        implied_volatility=0.5*(new_sigma+initial_sigma)
            
    return implied_volatility

S0=50
K=55
r=0.1
q=0.03
T=0.5
n=100

print("The implied volatility using bisection method and Black Scholes model: %.6f" % Bisection('BS', S0, K, r, q, T, 2.5, n, 'c', 'E', convergence=0.000001))
print("The implied volatility using bisection method and CRR model: %.6f" %Bisection('CRR', S0, K, r, q, T, 2.5, n, 'c', 'E', convergence=0.000001))
print("The implied volatility using bisection method and CRR model: %.6f" %Bisection('CRR', S0, K, r, q, T, 6.5, n, 'p', 'A', convergence=0.00001))

print("The implied volatility using Newton's method and Black Scholes model: %.6f" %Newton("BS", S0, K, r, q, T, 2.5, n, 'c', 'E', convergence=0.00000001))
print("The implied volatility using Newton's method and CRR model: %.6f" %Newton("CRR", S0, K, r, q, T, 2.5, n, 'c', 'E', convergence=0.00000001))
print("The implied volatility using Newton's method and CRR model: %.6f" %Newton('CRR', S0, K, r, q, T, 6.5, n, 'p', 'A', convergence=0.00000001))
