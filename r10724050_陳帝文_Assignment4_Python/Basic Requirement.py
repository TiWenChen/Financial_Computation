from turtle import st
import pandas as pd
import numpy as np
import scipy.stats

def lookback_maximum_put_option_CRR(St, r, q, sigma, t, T, S_init_max, n, otype):
    if otype=='E':
        otype=0
    elif otype=='A':
        otype=1
    u = np.exp(sigma * np.sqrt((T-t) / n))
    d = 1 / u
    P = (np.exp((r - q) * (T-t) / n) - d) / (u - d)

    CRR_price = np.zeros((n + 1, n + 1))
    CRR_price[0, 0] = St
    for i in range(1, n + 1):
        for j in range(i + 1):
            CRR_price[j, i] = round(CRR_price[0, 0] * (u ** (i - j)) * (d ** j), 10)

    #Adjust every nod's price

    for i in range(n - 2, -1, -1):
        for j in range(i + 1):
            CRR_price[j, i] = CRR_price[j + 1, i + 2]
    j = 0
    for i in range(0, n + 1, 2):
        CRR_price[j, i] = St
        j += 1

    #Smax matrix
    Smax = np.zeros((n + 1, n + 1, (n + 1) + n + 1))
    Smax[0, 0, 0] = S_init_max
    for i in range(1, n + 1):
        for j in range(i + 1):
            if j == 0: #最上層
                if CRR_price[j, i] >= Smax[j, i - 1, 0]:
                    Smax[j, i, 0] = CRR_price[j, i]
                else:
                    Smax[j, i, 0] = Smax[j, i - 1, 0]
            elif j == i: #最下層
                Smax[j, i, 0] = Smax[j - 1, i - 1, 0]
            else:
                count = 0
                for k in range(Smax.shape[-1]):  # 對準左邊
                    if Smax[j, i - 1, k] == 0:
                        break
                    if Smax[j, i - 1, k] >= CRR_price[j, i]:
                        if Smax[j, i - 1, k] not in Smax[j, i]:
                            Smax[j, i, count] = Smax[j, i - 1, k]
                            count += 1
                    elif Smax[j, i - 1, k] < CRR_price[j, i]:
                        if CRR_price[j, i] not in Smax[j, i]:  # 如果Smax比自身price小，且price還沒在Smax的list裡面
                            Smax[j, i, count] = CRR_price[j, i]
                            count += 1
                for l in range(Smax.shape[-1]):  # 對準左上
                    if Smax[j - 1, i - 1, l] == 0:
                        break
                    if Smax[j - 1, i - 1, l] >= CRR_price[j, i]:
                        if Smax[j - 1, i - 1, l] not in Smax[j, i]:
                            Smax[j, i, count] = Smax[j - 1, i - 1, l]
                            count += 1

                    elif Smax[j - 1, i - 1, l] < CRR_price[j, i]:
                        if CRR_price[j, i] not in Smax[j, i]:  # 如果Smax比自身price小，且price還沒在Smax的list裡面
                            Smax[j, i, count] = CRR_price[j, i]
                            count += 1
            sort = np.sort(Smax[j, i])
            Smax[j, i] = sort[::-1]

    #Intrinsic value
    IV_put = np.zeros(Smax.shape)
    for i in range(n + 1):
        for j in range(i + 1):
            for k in range(Smax.shape[-1]):
                if Smax[j, i, k] == 0:
                    break
                else:
                    IV_put[j, i, k] = Smax[j, i, k] - CRR_price[j, i]

    #put value
    put = np.zeros(IV_put.shape)
    put[:, -1, ] = IV_put[:, -1, ] #將最後一個column 抓下來
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            k1 = 0  # up 的index
            k2 = 0  # down 的index
            for k in range(Smax.shape[-1]):  # put 的index
                if Smax[j, i, k] == 0:
                    break

                if Smax[j, i, k] in Smax[j, i + 1]: #往右找的put value，且本nod 的Smax 出現在右邊nod 的Smax list裡
                    for m in range(k1, Smax.shape[-1]):
                        if Smax[j, i + 1, m] == Smax[j, i, k]:
                            A = put[j, i + 1, m]
                            break
                    k1 = m
                else:
                    for m in range(k1, Smax.shape[-1]): #往右找的put value，且本nod 的Smax 沒出現在右邊nod 的Smax list裡
                        if Smax[j, i + 1, m] == CRR_price[j, i + 1]:
                            A = put[j, i + 1, m]
                            break
                    k1 = m

                for M in range(k2, Smax.shape[-1]): #往下找的put value
                    if Smax[j + 1, i + 1, M] == Smax[j, i, k]:
                        B = put[j + 1, i + 1, M]
                        break
                k2 = M
                put[j, i, k] = max((P * A + (1 - P) * B)*np.exp(-r*(T-t)/n), otype*(IV_put[j, i, k]))
    #print("The lookback maximum put option value is: ", put[0, 0, 0])
    return put[0, 0, 0]

def lookback_maximum_put_option_Monte_Carlo(St, r, q, sigma, t, T, S_init_max, n, number_of_simulation, number_of_repetition):

    t=(T-t)/n #delta t
    payoff_list=[]
    for i in range(number_of_repetition):
        price_matrix = np.zeros((number_of_simulation, n + 1))
        price_matrix[..., 0] = St
        for j in range(1, n+1):
            delta_S=scipy.stats.norm.rvs(size=number_of_simulation, loc=(r-q-0.5*(sigma**2))*t, scale=sigma*np.sqrt(t))
            S=np.log(price_matrix[..., j-1])+delta_S
            price_matrix[..., j]=np.exp(S)  #reference Financial Computation Notes 2 page 5
        
        #for j in range(number_of_simulation):
            #plt.plot(price_matrix[j,...])
        #plt.show()
        
        #print(np.amax(price_matrix, axis=1)) #印出每個row 的max
        max_=np.maximum(S_init_max, np.amax(price_matrix, axis=1))
        payoff=np.maximum(max_-price_matrix[..., -1], 0)

        payoff_list.append(np.mean(np.exp(-r*(T-t))*payoff))
    upper_bond=np.mean(payoff_list)+ 2 * np.std(payoff_list)
    lower_bond = np.mean(payoff_list) - 2 * np.std(payoff_list)
    print("Value interval of the lookback maximum put option: %.6f ~ %.6f"%(lower_bond, upper_bond))


St=50
r=0.1
q=0
sigma=0.4
t=0
T=0.25
n=100
#S_init_max=60
otype='E'
number_of_simulation=10000
number_of_repetition=20


print('European put, Smax=50: ')
print(round(lookback_maximum_put_option_CRR(St, r, q, sigma, t, T, 50, n, 'E'),4))
print('\n')
print('European put, Smax=60: ')
print(round(lookback_maximum_put_option_CRR(St, r, q, sigma, t, T, 60, n, 'E'),4))
print('\n')
print('European put, Smax=70: ')
print(round(lookback_maximum_put_option_CRR(St, r, q, sigma, t, T, 70, n, 'E'), 4))
print('\n')
print('American put, Smax=50: ')
print(round(lookback_maximum_put_option_CRR(St, r, q, sigma, t, T, 50, n, 'A'), 4))
print('\n')
print('American put, Smax=60: ')
print(round(lookback_maximum_put_option_CRR(St, r, q, sigma, t, T, 60, n, 'A'), 4))
print('\n')
print('American put, Smax=70: ')
print(round(lookback_maximum_put_option_CRR(St, r, q, sigma, t, T, 70, n, 'A'), 4))

print('\n')
print('European Monte Carlo put, Smax=50: ')
E_50_m=lookback_maximum_put_option_Monte_Carlo(St, r, q, sigma, t, T, 50, n, number_of_simulation, number_of_repetition)
print('\n')
print('European Monte Carlo put, Smax=60: ')
E_60_m=lookback_maximum_put_option_Monte_Carlo(St, r, q, sigma, t, T, 60, n, number_of_simulation, number_of_repetition)
print('\n')
print('European Monte Carlo put, Smax=70: ')
E_70_m=lookback_maximum_put_option_Monte_Carlo(St, r, q, sigma, t, T, 70, n, number_of_simulation, number_of_repetition)

