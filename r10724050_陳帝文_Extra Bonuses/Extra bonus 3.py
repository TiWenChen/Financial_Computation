import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats


def Monte_Carlo_vanilla_american_option(S0, K, r, q, sigma, T, n, number_of_simulation, number_of_repetition, call):
    if call=='c':
        call=1
    elif call=='p':
        call=-1

    t=T/n
    value=[]
    for i in range(number_of_repetition):
        price_matrix=np.zeros((number_of_simulation, n+1))
        price_matrix[..., 0]=S0
        for j in range(1, n+1):
            delta_S=scipy.stats.norm.rvs(size=number_of_simulation, loc=(r-q-0.5*(sigma**2))*t, scale=sigma*np.sqrt(t))
            S=np.log(price_matrix[..., j-1])+delta_S
            price_matrix[..., j]=np.exp(S)

        value_matrix=np.zeros(price_matrix.shape)
        value_matrix[..., -1]=np.maximum(call*(price_matrix[..., -1]-K), 0)
        for j in range(n-1, -1, -1):
            EV=np.maximum(call*(price_matrix[..., j]-K), 0)
            effective_EV=[int(i) for i in (EV!=0)] #EV大於0的為1，當作指標
            HV=value_matrix[..., j+1]*np.exp(-r*t) #先將下一column的value全部更新成後一期的value折現，之後再將要early exercise的更新
            value_matrix[..., j]=HV
            effective_HV=HV*effective_EV
            effective_S=price_matrix[..., j]*effective_EV
            df=np.reshape(np.concatenate((effective_HV, effective_S, effective_S**2)), (number_of_simulation, 3), order='F') 
            #df=np.reshape(np.concatenate((effective_HV, effective_S, effective_S**2, effective_S**3)), (number_of_simulation, 4), order='F')
            df=df[~np.all(df==0, axis=1)] #將row全為0的，也就是被EV=0的，刪除
            X=np.delete(df, [0], axis=1) #X為第一個column後所有column
            Y=df[..., 0]
            try: #解決K=S0，column 0無法回歸的問題
                lm=LinearRegression()
                lm.fit(X, Y)

                x=np.reshape(np.concatenate(( effective_S, effective_S**2)), (number_of_simulation, 2), order='F') 
                #x=np.reshape(np.concatenate(( effective_S, effective_S**2, effective_S**3)), (number_of_simulation, 3), order='F') 
                expect_HV=lm.predict(x)*effective_EV #後面再乘一次effective_EV是為了消除row(EV==0)的intercept_
                
                for k, l in enumerate(expect_HV<EV):
                    if l==True:
                        value_matrix[k, j]==EV[k]
            except:
                value_matrix[..., j]=value_matrix[..., j+1]*np.exp(-r*t)
        value.append(np.mean(value_matrix[..., 0]))
    print(np.mean(value)-2*np.std(value)," ~ ", np.mean(value)+2*np.std(value))

def Monte_Carlo_lookback_American_option(S0, r, q, sigma, T, t, n, S_init_max, number_of_simulation, number_of_repetition, call='p'):
    if call=='c':
        call=1
    elif call=='p':
        call=-1

    value=[]
    delta_t=(T-t)/n
    for i in range(number_of_repetition):
        price_matrix=np.zeros((number_of_simulation, n+1))
        price_matrix[..., 0]=S0
        for j in range(1, n+1):
            delta_S=scipy.stats.norm.rvs(size=number_of_simulation, loc=(r-q-0.5*(sigma**2))*delta_t, scale=sigma*np.sqrt(delta_t))
            S=np.log(price_matrix[..., j-1])+delta_S
            price_matrix[..., j]=np.exp(S)
        
        value_matrix=np.zeros(price_matrix.shape)
        max_=np.maximum(S_init_max, np.amax(price_matrix, axis=1))
        value_matrix[..., -1]=np.maximum(call*(price_matrix[..., -1]-max_), 0)

        for j in range(n-1, -1, -1):
            max_=np.maximum(S_init_max, np.amax(price_matrix[..., 0:j+1], axis=1))
            EV=np.maximum(call*(price_matrix[..., j]-max_), 0)
            effective_EV=[int(i) for i in (EV!=0)] 
            HV=value_matrix[..., j+1]*np.exp(-r*delta_t) #先將下一column的value全部更新成後一期的value折現，之後再將要early exercise的更新
            value_matrix[..., j]=HV
            effective_HV=HV*effective_EV

            effective_S=price_matrix[..., j]*effective_EV
            df=np.reshape(np.concatenate((effective_HV, effective_S, effective_S**2)), (number_of_simulation, 3), order='F') 
            #df=np.reshape(np.concatenate((effective_HV, effective_S, effective_S**2, effective_S**3)), (number_of_simulation, 4), order='F')
            df=df[~np.all(df==0, axis=1)] #將row全為0的，也就是被EV=0的，刪除
            X=np.delete(df, [0], axis=1) #X為第一個column後所有column
            Y=df[..., 0]
            try:
                lm=LinearRegression()
                lm.fit(X, Y)

                x=np.reshape(np.concatenate(( effective_S, effective_S**2)), (number_of_simulation, 2), order='F') 
                #x=np.reshape(np.concatenate(( effective_S, effective_S**2, effective_S**3)), (number_of_simulation, 3), order='F') 
                expect_HV=lm.predict(x)*effective_EV #後面再乘一次effective_EV是為了消除row(EV==0)的intercept_
                    
                for k, l in enumerate(expect_HV<EV):
                    if l==True:
                        value_matrix[k, j]==EV[k]
            except:
                value_matrix[..., j]=value_matrix[..., j+1]*np.exp(-r*delta_t)
        value.append(np.mean(value_matrix[..., 0]))
    print(np.mean(value)-2*np.std(value)," ~ ", np.mean(value)+2*np.std(value))


def Monte_Carlo_arithmetic_American_call_option(S0, K, r, q, sigma, T, t, n, S_init_avg, number_of_simulation, number_of_repetition):

    value=[]
    delta_t=(T-t)/n
    for i in range(number_of_repetition):
        price_matrix=np.zeros((number_of_simulation, n+1))
        price_matrix[..., 0]=S0
        for j in range(1, n+1):
            delta_S=scipy.stats.norm.rvs(size=number_of_simulation, loc=(r-q-0.5*(sigma**2))*delta_t, scale=sigma*np.sqrt(delta_t))
            S=np.log(price_matrix[..., j-1])+delta_S
            price_matrix[..., j]=np.exp(S)
        
        value_matrix=np.zeros(price_matrix.shape)
        num_bef = int(1 + n * t / (T - t)) #S0前有幾期
        avg_=price_matrix[..., 1:].mean(axis=1)
        avg_= (num_bef*S_init_avg+n*avg_)/(num_bef+n)
        value_matrix[..., -1]=np.maximum(avg_-K, 0)

        for j in range(n-1, -1, -1):
            
            if j!=0:
                avg_=price_matrix[..., 1:(j+1)].mean(axis=1)
                avg_= (num_bef*S_init_avg+(j)*avg_)/(num_bef+j)
                EV=np.maximum(avg_-K, 0)
            else:
                avg_=np.array([S_init_avg]*number_of_simulation)
                EV=np.maximum(avg_-K, 0)

            effective_EV=[int(i) for i in (EV!=0)] 
            HV=value_matrix[..., j+1]*np.exp(-r*delta_t) #先將下一column的value全部更新成後一期的value折現，之後再將要early exercise的更新
            value_matrix[..., j]=HV
            effective_HV=HV*effective_EV

            effective_S=price_matrix[..., j]*effective_EV
            df=np.reshape(np.concatenate((effective_HV, effective_S, effective_S**2)), (number_of_simulation, 3), order='F') 
            #df=np.reshape(np.concatenate((effective_HV, effective_S, effective_S**2, effective_S**3)), (number_of_simulation, 4), order='F')
            df=df[~np.all(df==0, axis=1)] #將row全為0的，也就是被EV=0的，刪除
            X=np.delete(df, [0], axis=1) #X為第一個column後所有column
            Y=df[..., 0]
            
            try:
                lm=LinearRegression()
                lm.fit(X, Y)

                x=np.reshape(np.concatenate(( effective_S, effective_S**2)), (number_of_simulation, 2), order='F') 
                #x=np.reshape(np.concatenate(( effective_S, effective_S**2, effective_S**3)), (number_of_simulation, 3), order='F') 
                expect_HV=lm.predict(x)*effective_EV #後面再乘一次effective_EV是為了消除row(EV==0)的intercept_
                    
                for k, l in enumerate(expect_HV<EV):
                    if l==True:
                        value_matrix[k, j]==EV[k]
            except:
                value_matrix[..., j]=value_matrix[..., j+1]*np.exp(-r*delta_t)
        value.append(np.mean(value_matrix[..., 0]))
    print(np.mean(value)-2*np.std(value)," ~ ", np.mean(value)+2*np.std(value))

St=50
K=50
r=0.1
q=0.05
sigma=0.4
t=0
T=1
M=50
n=100
Savg_initial=60
number_of_simulation=10000
number_of_repetition=20
Monte_Carlo_vanilla_american_option(50, 50, 0.1, 0.05, 0.4, 0.5, 100, number_of_simulation, number_of_repetition, 'p')
Monte_Carlo_lookback_American_option(50, 0.1, 0, 0.4, 0.5, 0.25, 100, 60, number_of_simulation, number_of_repetition)
Monte_Carlo_arithmetic_American_call_option(50, 50, 0.1, 0.05, 0.8, 0.5, 0.25, 100, 50, number_of_simulation, number_of_repetition)