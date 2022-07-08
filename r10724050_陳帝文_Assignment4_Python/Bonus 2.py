import pandas as pd
import numpy as np
import scipy.stats

#Bonus 2
def Cheuk_Vorst_lookback_maximum_put_option(St, r, q, sigma, T, S_init_max, n, otype):
    if otype=='E':
        otype=0
    elif otype=='A':
        otype=1

    u = np.exp(sigma * np.sqrt(T / n))
    d=1/u
    mu=np.exp((r-q)*(T/n))

    price_matrix=np.zeros((n+1, n+1))
    price_matrix[-1, 0]=S_init_max/St
    for i in range(1, n+1):
        for j in range(i+1):
            column_index=-j-1
            price_matrix[column_index, i]=u**j

    put=np.zeros(price_matrix.shape)
    put[..., -1]=np.maximum(price_matrix[...,-1]-1, 0)
    P=(mu*u-1)/(mu*(u-d))
    P=1-P #??????????
    for i in range(n-1, -1, -1):
        for j in range(i+1):
            column_index=-j-1
            if j==0:
                put[column_index, i]=max(np.exp(-r*T/n)*(P*put[column_index-1, i+1]+(1-P)*put[column_index, i+1])*mu, otype*(price_matrix[column_index, i]-1))
            else:
                put[column_index, i]=max(np.exp(-r*T/n)*(P*put[column_index-1, i+1]+(1-P)*put[column_index+1, i+1])*mu, otype*(price_matrix[column_index, i]-1))
    value=put[-1, 0]*St
    #print("The lookback maximum put option value is: ", value)
    return value

St=50
r=0.1
q=0
sigma=0.4
t=0
T=0.25
n=1000
S_init_max=50
otype='A'
print('European put: ')
print(round(Cheuk_Vorst_lookback_maximum_put_option(St, r, q, sigma, T, S_init_max, n, 'E'), 4))
print('American put: ')
print(round(Cheuk_Vorst_lookback_maximum_put_option(St, r, q, sigma, T, S_init_max, n, 'A'), 4))

