import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# PUTS USING CHEBYSHEV POLYNOMIALS 
N = 1000 # number of time steps, number of discretization points, time points that the option can be exercised
M = 1000 # number of paths of the underlying asset
BF = 20 # number of basis functions used, pandas cannot handle more than 20
T = 1 # time from 'now' to expiration of option in years 
dt = T/N # time between each time step 

PRICE = 1
STRIKE_PRICE = 1

# initialize table with stock price
TABLE = np.zeros((M, N + 1))
TABLE[:, 0] = PRICE

# Filling the table with stochastic diferential equation 
r = 0.06 # risk free rate (%)
sigma = 0.3 # constant for eq
# Z is standard Brownian Motion 
for i in range(1, N + 1):
    dZ = np.random.normal(size=M) * np.sqrt(dt) 
    # see eq 7 in longstaff-shcwartz 2001
    dS = TABLE[:, i - 1] * ((r * dt) + (sigma * dZ)) 
    TABLE[:, i] = TABLE[:, i -1] + dS

TABLE_DF = pd.DataFrame(TABLE)

mu = 0.06

# "brownian motion" graph 
TABLE_DF.transpose().plot(color="red", alpha=0.3)
plt.legend([])
plt.plot([0,N], [STRIKE_PRICE, STRIKE_PRICE], label="strike price")
plt.show()

# determining in-the-money 
Y = (STRIKE_PRICE - TABLE_DF[N]).map(lambda v: max(v, 0))
X = TABLE_DF[N-1]
IN_THE_MONEY = X < STRIKE_PRICE
X = X[IN_THE_MONEY]
Y = Y[IN_THE_MONEY]

# scatter plot
plt.scatter(X, Y)
plt.show()

# set basis functions 
# TODO: SETTING BASIS FUNCTIONS 
poly = pd.DataFrame(index=X.index)
poly[0] = 1
poly[1] = X
poly[2] = (2 * (X ** 2)) - 1
poly[3] = (4 * (X ** 3)) - (3 * X)
poly[4] = (8 * (X ** 4)) - (8 * (X ** 2)) + 1
poly[5] = (16 * (X ** 5)) - (20 * (X ** 3)) + (5 * X)


# stats models 
model = sm.OLS(Y, poly)
res = model.fit()
coef = res.params

# continuing / exercise 
continuation = (poly * coef).sum(axis=1)
exercise = (STRIKE_PRICE - TABLE_DF[N-1][IN_THE_MONEY])

# x scale 
x = np.linspace(.5, 1.5, 100)

# basis functions on y 
# TODO: SETTING BASIS FUNCTIONS 
y = 1 * coef[0] + x * coef[1] + ((2 * (x ** 2)) - 1) * coef[2] + ((4 * (X ** 3)) - (3 * X)) * coef[3] + ((8 * (X ** 4)) - (8 * (X ** 2)) + 1) * coef[4] + ((16 * (X ** 5)) - (20 * (X ** 3)) + (5 * X)) * coef[5]

# plotting 
plt.figure(figsize=(10,10))
plt.plot(x,y, linestyle=":", color="blue")
plt.xlabel(f"price at {N-1}", fontdict=None, labelpad=None, loc=None)
plt.ylabel(f"amount profit at {N}", fontdict=None, labelpad=None, loc=None)
plt.title(f'Puts with starting price: {PRICE} and strike price: {STRIKE_PRICE}')
plt.scatter(X, Y, color="red", label="Y (discounted exercise later, actual value that was gotten later)")
plt.scatter(X, continuation, label="continuation, theoretical", marker="x", color="blue")
plt.scatter(X, exercise, label="exercise now", marker="+", color="green")
plt.legend()
plt.show()
