import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

St=50
K=50
r=0.1
q=0.05
sigma=0.8
t=0
T=0.25
M=100
n=100
otype='E'
Savg_initial=50



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
    
    return call_matrix[0, 0, 0]

def Arithmetic_average_call_CRR_log(St, K, r, q, sigma, t, T, M, n, Savg_initial, otype):
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
            price_matrix[j, i] = price_matrix[0, 0] * (u ** (i - j)) * (d ** j)

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
                #average_matrix[j, i, 0] = Amax
            #else:
                # average_list
            for k in range(M + 1):
                    #Different from the original method
                average_matrix[j, i, k] =round(np.exp ((M - k) * np.log(Amax) / M + k * np.log(Amin) / M), 10)

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
                        for m1 in range(k1, M+1):
                            if Au == average_matrix[j, i + 1, m1]:
                                Cu = call_matrix[j, i + 1, m1]
                                break
                        k1 = m1
                    else:
                        for m1 in range(k1, M+1):
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
                        for m2 in range(k2, M+1):
                            if Ad == average_matrix[j + 1, i + 1, m2]:
                                Cd = call_matrix[j + 1, i + 1, m2]
                                break
                        k2 = m2
                    else:
                        for m2 in range(k2, M+1):
                            if Ad > average_matrix[j + 1, i + 1, m2]:
                                wd = (average_matrix[j + 1, i + 1, m2 - 1] - Ad) / (
                                        average_matrix[j + 1, i + 1, m2 - 1] - average_matrix[j + 1, i + 1, m2])
                                break
                        Cd = wd * call_matrix[j + 1, i + 1, m2] + (1 - wd) * call_matrix[j + 1, 1 + i, m2 - 1]
                        k2 = m2

                call_matrix[j, i, k] = max((P * Cu + (1 - P) * Cd) * np.exp(-r * (T - t) / n), otype * (IV_matrix[j, i, k]))

    #print("The logarithmically CRR arthimetic average call value: ", round(call_matrix[0, 0, 0],4))
    return call_matrix[0, 0, 0]

df=pd.DataFrame()

for i in range(1, 9):
    a=Arithmetic_average_call_CRR(St, K, r, q, sigma, t, T, 50*i, n, Savg_initial, otype)
    b=Arithmetic_average_call_CRR_log(St, K, r, q, sigma, t, T, 50*i, n, Savg_initial, otype)
    df['M: %d'%(i*50)]=[a, b]
df.index=['Linearly', 'Logarithmically']
print(df)
df.to_excel("Bonus_1.xlsx")

plt.figure(figsize=(16,8))
plt.plot([50*i for i in range(1, 9)], df.iloc[0], label='Linearly')
plt.plot([50*i for i in range(1, 9)], df.iloc[1], label='Logarithmically')
plt.grid(True)
plt.legend()
plt.show()