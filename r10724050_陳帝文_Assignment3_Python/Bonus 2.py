import pandas as pd
import numpy as np
import scipy.stats

def Inversed_Cholesky(num_of_series, samplings=10000):
    Z=[]
    for i in range(num_of_series):
        #np.random.seed(seed=50+i)
        z=scipy.stats.norm.rvs(size=samplings, loc=0, scale=1)
        z=z-np.mean(z)
        Z.append(z)
    A=Cholesky_decom(np.cov(Z))
    #A=Eigen_decom(np.cov(Z))
    A_inv=np.linalg.inv(A)
    Sz=np.dot(np.array(Z).T, A_inv)
    #plt.figure(figsize=(8,6))
    #sn.heatmap(np.cov(Sz.T), annot=True)
    #plt.show()
    return Sz

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


def Rainbow_option_inverse_Cholesky(K, r, T, number_of_simulation, number_of_repetition, number_of_underlying_assets, S, q, sigma, correlation):
    #construct covaraince matrix from the data givin
    cov_matrix=covmatrix(T, number_of_underlying_assets, sigma, correlation)


    #Do the Cholesky decomposition
    A_hat=Cholesky_decom(cov_matrix)

    #sampling
    payoff=[]
    for i in range(number_of_repetition):

        Sz=Inversed_Cholesky(number_of_underlying_assets, number_of_simulation)
        #sample 完 covariance matrix 為 identity matrix 的 n 組樣本
        
        #轉換成給定 covariance matrix 的 n 組樣本
        Sr=np.dot(np.array(Sz),A_hat)
        
        '''
        #Show the covariance heatmap
        df=pd.DataFrame(Sr)
        sn.heatmap(df.cov(),annot=True)
        plt.show()
        '''
        
        for j in range(number_of_underlying_assets):
            #print(np.mean(Sr[..., j]), np.var(Sr[...,j]))
            Sr[..., j]=np.exp(Sr[...,j]+np.log(S[j])+(r-q[j]-0.5*(sigma[j]**2))*T)
            
        #display(pd.DataFrame(Sr))
        '''
        Sr_mean=[]
        for j in range(number_of_underlying_assets):
            Sr_mean.append(np.mean(Sr[...,j]))
            #print(np.mean(Sr[...,j]))
        payoff.append(max(max(Sr_mean)-K,0))
        '''
        Sr_max=[]
        for j in range(number_of_simulation):
            Sr_max.append(np.max(Sr[j,...])) #將每n檔模擬出來的股票先找出最大者
        #print((np.array(Sr_max)-K))
        payoff.append(np.exp(-r*T)*np.mean(np.maximum(np.array(Sr_max)-K, 0)))
        #display(pd.DataFrame(payoff).T)

    mean=np.mean(payoff)
    se=np.std(payoff)
    upper_bond=mean+2*se
    lower_bond=mean-2*se
    #print(mean, se)
    print('The confidence interval of the Rainbow Option value: %.8f ~ %.8f'%(lower_bond,upper_bond))

K=100
T=0.5
r=0.1
number_of_simulation=10000
number_of_repetition=20
number_of_underlying_assets=5
S=[95, 95, 95, 95, 95]
q=[0.05, 0.05, 0.05, 0.05, 0.05]
sigma=[0.5, 0.5, 0.5, 0.5, 0.5]
correlation=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
Rainbow_option_inverse_Cholesky(K, r, T, number_of_simulation, number_of_repetition, number_of_underlying_assets, S, q, sigma, correlation)
