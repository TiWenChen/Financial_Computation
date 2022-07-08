import pandas as pd
import numpy as np
import scipy.stats




St=50
r=0.1
q=0
sigma=0.4
t=0
T=0.25
n=1000
S_init_max=60
otype='E'

#Bonus 1
#Determine Smax at every node
def lookback_option_bonus(St, r, q, sigma, t, T, S_init_max, n, otype):
    if otype=='E':
        otype=0
    elif otype=='A':
        otype=1
    price_matrix=np.zeros((n+1, n+1))
    price_matrix[0, 0]=St
    u = np.exp(sigma * np.sqrt((T-t) / n))
    d = 1 / u
    P = (np.exp((r - q) * (T-t) / n) - d) / (u - d)
    for i in range(1, n+1):
        for j in range(i+1):
            price_matrix[j, i]=round(price_matrix[0, 0] * (u ** (i - j)) * (d ** j), 10)

    for i in range(1, n+1):
        for j in range(i+1):
            '''
            if j==(i-j):
                price_matrix[j, i]=St #將u和d次方數一樣多的直接設St，避免後方有小數點程式判斷錯誤
            else:
            '''
            price_matrix[j, i]=price_matrix[0, 0] * (u ** (i - j)) * (d ** j)

    #Adjust every nod's price

    for i in range(n - 2, -1, -1):
        for j in range(i + 1):
            price_matrix[j, i] = price_matrix[j + 1, i + 2]
    j = 0
    for i in range(0, n + 1, 2):
        price_matrix[j, i] = St
        j += 1

    #看最後一個column每個點擁有幾個Smax值，並紀錄最多者
    Smax_max_len=0
    #print(price_matrix)
    for j in range(n+1): #用j代表每個nodes有幾個d
        i=n-j
        Smax_list=[]
        for k in range(j+1):
            #print(price_matrix[k, i+k], k, j, i)
            if price_matrix[k, i+k]>=S_init_max:
                Smax_list.append(price_matrix[k, i+k])
            else:
                if S_init_max not in Smax_list:
                    Smax_list.append(S_init_max)
        if len(Smax_list)>Smax_max_len:
            Smax_max_len=len(Smax_list)


    Smax_matrix=np.zeros((n+1, n+1, Smax_max_len))
    Smax_matrix[0, 0, 0]=S_init_max
    for i in range(1, n+1):
        for j in range(i+1):
            
            I=i-j
            for k in range(j+1):
                if price_matrix[k, I+k]>=S_init_max:
                    Smax_matrix[j, i, k]=price_matrix[k, I+k]
                else:
                    if S_init_max not in Smax_matrix[j, i]:
                        Smax_matrix[j, i, k]=S_init_max
                

    #Intrinsic value
    IV_put = np.zeros(Smax_matrix.shape)
    for i in range(n + 1):
        for j in range(i + 1):
            for k in range(Smax_matrix.shape[-1]):
                if Smax_matrix[j, i, k] == 0:
                    break
                else:
                    IV_put[j, i, k] = Smax_matrix[j, i, k] - price_matrix[j, i]

    #put value
    put = np.zeros(IV_put.shape)
    put[:, -1, ] = IV_put[:, -1, ] #將最後一個column 抓下來
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            k1 = 0  # up 的index
            k2 = 0  # down 的index
            for k in range(Smax_matrix.shape[-1]):  # put 的index
                if Smax_matrix[j, i, k] == 0:
                    break

                if Smax_matrix[j, i, k] in Smax_matrix[j, i + 1]: #往右找的put value，且本nod 的Smax 出現在右邊nod 的Smax list裡
                    for m in range(k1, Smax_matrix.shape[-1]):
                        if Smax_matrix[j, i + 1, m] == Smax_matrix[j, i, k]:
                            A = put[j, i + 1, m]
                            break
                    k1 = m
                else:
                    for m in range(k1, Smax_matrix.shape[-1]): #往右找的put value，且本nod 的Smax 沒出現在右邊nod 的Smax list裡
                        if Smax_matrix[j, i + 1, m] == price_matrix[j, i + 1]:
                            A = put[j, i + 1, m]
                            break
                    k1 = m

                for M in range(k2, Smax_matrix.shape[-1]): #往下找的put value
                    if Smax_matrix[j + 1, i + 1, M] == Smax_matrix[j, i, k]:
                        B = put[j + 1, i + 1, M]
                        break
                k2 = M
                put[j, i, k] = max((P * A + (1 - P) * B)*np.exp(-r*(T-t)/n), otype*(IV_put[j, i, k]))
    #print("The lookback maximum put option value is: ", put[0, 0, 0])
    return put[0, 0, 0]

print('European put, Smax=50: ')
print(round(lookback_option_bonus(St, r, q, sigma, t, T, 50, n, 'E'),4))
'''
print('\n')
print('European put, Smax=60: ')
print(round(lookback_option_bonus(St, r, q, sigma, t, T, 60, n, 'E'),4))
print('\n')
print('European put, Smax=70: ')
print(round(lookback_option_bonus(St, r, q, sigma, t, T, 70, n, 'E'), 4))
print('\n')
print('American put, Smax=50: ')
print(round(lookback_option_bonus(St, r, q, sigma, t, T, 50, n, 'A'), 4))
print('\n')
print('American put, Smax=60: ')
print(round(lookback_option_bonus(St, r, q, sigma, t, T, 60, n, 'A'), 4))
print('\n')
print('American put, Smax=70: ')
print(round(lookback_option_bonus(St, r, q, sigma, t, T, 70, n, 'A'), 4))

'''