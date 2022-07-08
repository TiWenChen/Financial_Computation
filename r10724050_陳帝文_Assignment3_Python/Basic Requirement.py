import pandas as pd
import numpy as np
import scipy.stats
import random

def covmatrix(T, number_of_underlying_assets, sigma, correlation):
        
    #construct covaraince matrix from the data givin

    cov_matrix=np.zeros([number_of_underlying_assets,number_of_underlying_assets])
    corr_count=0
    for i, k in enumerate(cov_matrix):
        for j, l in enumerate(k):
            if i==j:
                cov_matrix[i][j]=(sigma[i]**2)*T
            elif j>i:
                cov_matrix[i][j]=sigma[i]*sigma[j]*T*correlation[corr_count]
                corr_count+=1
    for i, k in enumerate(cov_matrix):
        for j, l in enumerate(k):
            if i>j:
                cov_matrix[i][j]=cov_matrix[j][i]
    return cov_matrix

def Cholesky_decom(covmatrix):
    A=np.zeros([covmatrix.shape[0],covmatrix.shape[1]])
    for i,k in enumerate(A):
        for j,l in enumerate(k):

            if i==0 and j==0:
                A[i][j]=np.sqrt(covmatrix[i][j])

            elif i==0 and j!=0:
                A[i][j]=covmatrix[i][j]/A[0][0]

            elif i==j and i!=0 and j!=0:
                A[i][j]=np.sqrt(covmatrix[i][j]-sum([A[k][i]**2 for k in range(i)]))

            elif j>i and i!=0:
                a_k_i=[A[k][i] for k in range(i)]
                a_k_j=[A[k][j] for k in range(i)]
                a_m=[a_k_j[k]*a_k_i[k] for k in range(len(a_k_i))]
                A[i][j]=(covmatrix[i][j]-sum(a_m))/A[i][i]
    return A

'''
def Rainbow_option(K, r, T, number_of_simulation, number_of_repetition, number_of_underlying_assets, S, q, sigma, correlation):
    #S, q, sigma, correlation 都是以 list 輸入
    #sigma=[sigma_1, sigma_2, sigma_3....]
    #correlation=[rho_12, rho_13, rho_14, ..., rho_23, rho_24,..., rho_34, ...]

    #construct covaraince matrix from the data givin
    cov_matrix=covmatrix(T, number_of_underlying_assets, sigma, correlation)


    #Do the Cholesky decomposition
    A=Cholesky_decom(cov_matrix)

    #sampling
    payoff=[]
    for i in range(number_of_repetition):
        Sz=[]
        for j in range(number_of_underlying_assets):
            #np.random.seed(seed=50+j)
            Sz.append(scipy.stats.norm.rvs(size=number_of_simulation, loc=0, scale=1))

        Sr=np.dot(np.array(Sz).T,A)
        
        #Show the covariance heatmap
        #df=pd.DataFrame(Sr)
        #sn.heatmap(df.cov(),annot=True)
        #plt.show()
        
        for j in range(number_of_underlying_assets):
            Sr[...,j]=np.exp(Sr[...,j]+np.log(S[j])+(r-q[j]-0.5*(sigma[j]**2))*T)
            
            
        #display(pd.DataFrame(Sr))
        #Show the Covariance Matrix

        #display(df.T)

        
        Sr_max=[]
        for j in range(number_of_simulation):
            Sr_max.append(np.max(Sr[j,...])) #將每n檔模擬出來的股票先找出最大者
        payoff.append(np.exp(-r*T)*np.mean(np.maximum(np.array(Sr_max)-K, 0)))

    mean=np.mean(payoff)
    se=np.std(payoff)
    upper_bond=mean+2*se
    lower_bond=mean-2*se
    print('The confidence interval of the Rainbow Option value: %.4f ~ %.4f'%(lower_bond,upper_bond))
    print("The midpoint of the confidence interval: %.4f"%mean)
'''

def Rainbow_option(K, r, T, number_of_simulation, number_of_repetition, number_of_underlying_assets, S, q, sigma, correlation):
    #S, q, sigma, correlation 都是以 list 輸入
    #sigma=[sigma_1, sigma_2, sigma_3....]
    #correlation=[rho_12, rho_13, rho_14, ..., rho_23, rho_24,..., rho_34, ...]

    #construct covaraince matrix from the data givin
    cov_matrix=covmatrix(T, number_of_underlying_assets, sigma, correlation)


    #Do the Cholesky decomposition
    A=Cholesky_decom(cov_matrix)

    #sampling
    payoff=[]
    payoff_r=[]
    payoff_in=[]
    for i in range(number_of_repetition):
        Sz=[]
        Sz_inverse=[]
        for j in range(number_of_underlying_assets):
            #np.random.seed(seed=50+j)
            z=scipy.stats.norm.rvs(size=number_of_simulation, loc=0, scale=1)
            Sz.append(z)
            z=z-np.mean(z)
            Sz_inverse.append(z)


        variance_reduct_Sz=np.array(Sz)
        variance_reduct_Sz=variance_reduct_Sz[...,0:int(0.5*np.array(Sz).shape[1])]
        reverse_Sz=(-1)*variance_reduct_Sz

        variance_reduct_Sz=np.concatenate((variance_reduct_Sz, reverse_Sz),axis=1)

        a=Cholesky_decom(np.cov(Sz_inverse))
        A_inv=np.linalg.inv(a)
        Sz_inverse=np.dot(np.array(Sz_inverse).T, A_inv)

        Sr=np.dot(np.array(Sz).T,A)
        Sr_r=np.dot(np.array(variance_reduct_Sz).T,A)
        Sr_in=np.dot(np.array(Sz_inverse),A)

        '''
        #Show the covariance heatmap
        df=pd.DataFrame(Sr)
        sn.heatmap(df.cov(),annot=True)
        plt.show()
        '''
        for j in range(number_of_underlying_assets):
            Sr[...,j]=np.exp(Sr[...,j]+np.log(S[j])+(r-q[j]-0.5*(sigma[j]**2))*T)
        
        for j in range(number_of_underlying_assets):
            Sr_r[...,j]=np.exp(Sr_r[...,j]+np.log(S[j])+(r-q[j]-0.5*(sigma[j]**2))*T)
        
        for j in range(number_of_underlying_assets):
            Sr_in[...,j]=np.exp(Sr_in[...,j]+np.log(S[j])+(r-q[j]-0.5*(sigma[j]**2))*T)
        #display(pd.DataFrame(Sr))
        #Show the Covariance Matrix

        #display(df.T)

        
        Sr_max=[]
        for j in range(number_of_simulation):
            Sr_max.append(np.max(Sr[j,...])) #將每n檔模擬出來的股票先找出最大者
        payoff.append(np.exp(-r*T)*np.mean(np.maximum(np.array(Sr_max)-K, 0)))

        Sr_r_max=[]
        for j in range(number_of_simulation):
            Sr_r_max.append(np.max(Sr_r[j,...])) #將每n檔模擬出來的股票先找出最大者
        payoff_r.append(np.exp(-r*T)*np.mean(np.maximum(np.array(Sr_r_max)-K, 0)))

        Sr_in_max=[]
        for j in range(number_of_simulation):
            Sr_in_max.append(np.max(Sr_in[j,...])) #將每n檔模擬出來的股票先找出最大者
        payoff_in.append(np.exp(-r*T)*np.mean(np.maximum(np.array(Sr_in_max)-K, 0)))

    mean=np.mean(payoff)
    se=np.std(payoff)
    upper_bond=mean+2*se
    lower_bond=mean-2*se
    print('The confidence interval of the rainbow option value: %.4f ~ %.4f'%(lower_bond,upper_bond))
    print("The midpoint of the confidence interval: %.4f"%mean)

    mean_r=np.mean(payoff_r)
    se_r=np.std(payoff_r)
    upper_bond_r=mean_r+2*se_r
    lower_bond_r=mean_r-2*se_r
    print('The confidence interval of the variance reduction rainbow option value: %.4f ~ %.4f'%(lower_bond_r,upper_bond_r))
    print("The midpoint of the confidence interval: %.4f"%mean_r)

    mean_in=np.mean(payoff_in)
    se_in=np.std(payoff_in)
    upper_bond_in=mean_in+2*se_in
    lower_bond_in=mean_in-2*se_in
    print('The confidence interval of the inversed Cholesky rainbow option value: %.4f ~ %.4f'%(lower_bond_in,upper_bond_in))
    print("The midpoint of the confidence interval: %.4f"%mean_in)

K=100
T=0.5
r=0.1
number_of_simulation=10000
number_of_repetition=20

number_of_underlying_assets=2
S=[95, 95]
q=[0.05, 0.05]
sigma=[0.5, 0.5]
correlation=[1]
Rainbow_option(K, r, T, number_of_simulation, number_of_repetition, number_of_underlying_assets, S, q, sigma, correlation)
print("=================================================")

number_of_underlying_assets=2
S=[95, 95]
q=[0.05, 0.05]
sigma=[0.5, 0.5]
correlation=[-1]
Rainbow_option(K, r, T, number_of_simulation, number_of_repetition, number_of_underlying_assets, S, q, sigma, correlation)
print("=================================================")

number_of_underlying_assets=5
S=[95, 95, 95, 95, 95]
q=[0.05, 0.05, 0.05, 0.05, 0.05]
sigma=[0.5, 0.5, 0.5, 0.5, 0.5]
correlation=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
Rainbow_option(K, r, T, number_of_simulation, number_of_repetition, number_of_underlying_assets, S, q, sigma, correlation)