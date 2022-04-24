import numpy
# import pandas as pd
# import matplotlib.pyplot as plt

N = 10 # number of time steps, number of discretization points, time points that the option can be exercised
M = 200 # number of paths of the underlying asset
T = 1 # time from 'now' to expiration of option in years 
dt = T/N # time between each time step 
r = 0.06 # risk free rate (%)
S = 1 # stock price 
sigma = 0.3
mu = 0.06

TABLE = numpy.zeros((M, N + 1))
TABLE[:, 0] = S

# filling the table standard Brownian motion
for i in range(1, N + 1):
    dZ = numpy.random.normal(size=M) * numpy.sqrt(dt) 
    # see eq 7 in longstaff-shcwartz 2001
    dS = TABLE[:, i - 1] * ((r * dt) + (sigma * dZ)) 
    TABLE[:, i] = TABLE[:, i -1] + dS

print(TABLE)


# df = pd.DataFrame(TABLE)
k = 1.1
# df.transpose().plot(color="red", alpha=0.3)
# plt.legend([])
# plt.plot([0,N], [k, k], label="strke price")
# plt.show()

# df
# discounting the payoff
Y = []
for i in range(0, N):
    Y.append(max(k - TABLE[N][i], 0))
    
print(Y)
SK = 1.05 # strike price 
IN_THE_MONEY = TABLE > SK
# print(IN_THE_MONEY)

# check if continuing is more profitable than exercising that that point
