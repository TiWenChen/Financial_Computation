import pandas as pd
import numpy as np
import scipy.stats

St=50
K=50
r=0.1
q=0.05
sigma=0.8
t=0
T=0.25
M=100
n=100
otype='A'
Savg_initial=50
number_of_simulation=10000
number_of_repetition=20


def Arithmetic_average_call_CRR(St, K, r, q, sigma, t, T, M, n, Savg_initial, otype):
    price_matrix = np.zeros((n + 1, n + 1))
    price_matrix[0, 0] = St

    if otype == 'E':
        otype = 0
    elif otype == 'A':
        otype = 1
    u = np.exp(sigma * np.sqrt((T - t) / n))
    d = 1 / u
    P = (np.exp((r - q) * (T - t) / n) - d) / (u - d)

    # CRR binomial tree
    for i in range(1, n + 1, 1):
        for j in range(i + 1):
            price_matrix[j, i] = round(price_matrix[0, 0] * (u ** (i - j)) * (d ** j), 10)

    # Adjusted node's price
    for i in range(n - 2, -1, -1):
        for j in range(i + 1):
            price_matrix[j, i] = price_matrix[j + 1, i + 2]
    j = 0
    for i in range(0, n + 1, 2):
        price_matrix[j, i] = St
        j += 1

    # Arithmetric matrix
    average_matrix = np.zeros((n + 1, n + 1, M + 1))
    average_matrix[0, 0, 0] = Savg_initial
    # num_bef: St 前期數
    num_bef = int(1 + n * t / (T - t))

    for i in range(1, n + 1):
        for j in range(i + 1):
            # Amax
            # u_sum: u+u^2+...+u^(i-j)
            u_sum = np.array([u ** k for k in range(1, i - j + 1)])
            u_sum = np.sum(u_sum)

            # d_sum
            d_sum = np.array([u ** (i - j) * (d ** k) for k in range(1, j + 1)])
            d_sum = np.sum(d_sum)

            Amax = (Savg_initial * num_bef + St * u_sum + St * d_sum) / (num_bef + i)

            # Amin
            # d_sum
            d_sum = np.array([d ** k for k in range(1, j + 1)])
            d_sum = np.sum(d_sum)

            # u_sum
            u_sum = np.array([(u ** k) * (d ** j) for k in range(1, i - j + 1)])
            u_sum = np.sum(u_sum)

            Amin = (Savg_initial * num_bef + St * d_sum + St * u_sum) / (num_bef + i)

            #if Amax == Amin:
                #average_matrix[j, i, 0] = Amax #########################################################
            #else:
                #average_list
            for k in range(M+1):
                average_matrix[j, i, k]= round((M-k)*Amax/M+k*Amin/M, 10)  ########

    # intrinsic value matrix
    IV_matrix = np.zeros(average_matrix.shape)
    for i in range(n + 1):
        for j in range(1 + i):
            IV_matrix[j, i] = np.maximum(average_matrix[j, i] - K, 0)

    # call value
    call_matrix = np.zeros(IV_matrix.shape)
    call_matrix[:, -1, ] = IV_matrix[:, -1, ]


    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            k1 = 0  # up 的 index
            k2 = 0  # down 的 index
            for k in range(M+1):
                Au = round(((num_bef + i) * average_matrix[j, i, k] + price_matrix[j, i + 1]) / (num_bef + i + 1), 10)
                Ad = round(((num_bef + i) * average_matrix[j, i, k] + price_matrix[j + 1, i + 1]) / (num_bef + i + 1), 10)
                if average_matrix[j, i, k] == 0:
                    break

                if j == 0:  # 最上層一定只有一個 call value 可能
                    Cu = call_matrix[j, i + 1, 0]
                else:
                    if Au in average_matrix[j, i + 1]:
                        for m1 in range(k1, M+1): ############################  k1
                            if Au == average_matrix[j, i + 1, m1]:
                                Cu = call_matrix[j, i + 1, m1]
                                break
                        k1 = m1
                    else:
                        for m1 in range(k1, M+1): ############################  k1
                            if Au > average_matrix[j, i + 1, m1]:
                                wu = (average_matrix[j, i + 1, m1 - 1] - Au) / (
                                            average_matrix[j, i + 1, m1 - 1] - average_matrix[j, i + 1, m1])
                                break
                        Cu = wu * call_matrix[j, i + 1, m1] + (1 - wu) * call_matrix[j, 1 + i, m1 - 1]
                        k1 = m1

                if j == i:
                    Cd = call_matrix[j + 1, i + 1, 0]
                else:
                    if Ad in average_matrix[j + 1, i + 1]:
                        for m2 in range(k2, M+1): ###########################  k2
                            if Ad == average_matrix[j + 1, i + 1, m2]:
                                Cd = call_matrix[j + 1, i + 1, m2]
                                break
                        k2 = m2
                    else:
                        for m2 in range(k2, M+1): ##########################  k2
                            if Ad > average_matrix[j + 1, i + 1, m2]:
                                wd = (average_matrix[j + 1, i + 1, m2 - 1] - Ad) / (
                                            average_matrix[j + 1, i + 1, m2 - 1] - average_matrix[j + 1, i + 1, m2])
                                break
                        Cd = wd * call_matrix[j + 1, i + 1, m2] + (1 - wd) * call_matrix[j + 1, 1 + i, m2 - 1]
                        k2 = m2

                call_matrix[j, i, k] = max((P * Cu + (1 - P) * Cd) * np.exp(-r * (T - t) / n), otype * (IV_matrix[j, i, k]))
    #print("The CRR arthimetic average call value: ", round(call_matrix[0, 0, 0],4))
    
    return call_matrix[0, 1, 0], call_matrix[1, 1, 0], call_matrix[0, 0, 0]
#print(Arithmetic_average_call_CRR(St, K, r, q, sigma, t, T, M, n, Savg_initial, 'E'))
#Arithmetic_average_call_CRR(St, K, r, q, sigma, t, T, M, n, Savg_initial, 'A')
print(Arithmetic_average_call_CRR(St, K, r, q, sigma, 0.25, 0.5, M, n, Savg_initial, 'E'))
#Arithmetic_average_call_CRR(St, K, r, q, sigma, 0.25, 0.5, M, n, Savg_initial, 'A')


def Arithmetic_average_call_Monte_Carlo(St, K, r, q, sigma, t, T, M, n, Savg_initial, number_of_simulation, number_of_repetition):

    delta_t = (T-t) / n  # delta t
    payoff_list = []
    for i in range(number_of_repetition):
        price_matrix = np.zeros((number_of_simulation, n + 1))
        price_matrix[..., 0] = St
        for j in range(1, n+1):
            delta_S = scipy.stats.norm.rvs(size=number_of_simulation, loc=(r-q-0.5*(sigma**2))*delta_t, scale=sigma*np.sqrt(delta_t))
            S = np.log(price_matrix[..., j-1])+delta_S
            price_matrix[..., j]=np.exp(S)

        num_bef = int(1 + n * t / (T - t))
        avg_=price_matrix[..., 1:].mean(axis=1)
        avg_= (num_bef*Savg_initial+n*avg_)/(num_bef+n)
        payoff=np.maximum(avg_-K, 0)

        payoff_list.append(np.mean(np.exp(-r * (T-t)) * payoff))
    upper_bond=np.mean(payoff_list)+ 2 * np.std(payoff_list)
    lower_bond = np.mean(payoff_list) - 2 * np.std(payoff_list)
    print("The Monre Carlo value interval of the arithmetic average call: %.6f ~ %.6f"%(lower_bond, upper_bond))

#Arithmetic_average_call_Monte_Carlo(St, K, r, q, sigma, 0, 0.25, M, n, Savg_initial, number_of_simulation, number_of_repetition)
#Arithmetic_average_call_Monte_Carlo(St, K, r, q, sigma, 0.25, 0.5, M, n, Savg_initial, number_of_simulation, number_of_repetition)
