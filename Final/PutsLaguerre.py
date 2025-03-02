import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# PUTS USING LAGUERRE POLYNOMIALS
N = 10 # number of time steps, number of discretization points, time points that the option can be exercised
M = 20 # number of paths of the underlying asset
BF = 20 # number of basis functions used, pandas cannot handle more than 20
T = 1 # time from 'now' to expiration of option in years 
dt = T/N # time between each time step 

PRICE = 1
STRIKE_PRICE = 1.08

# initialize table with stock price
TABLE = np.zeros((M, N + 1))
TABLE[:, 0] = PRICE

# Filling the table with stochastic diferential equation 
r = 0.06 # risk free rate (%)
sigma = 0.43 # constant for eq
# Z is standard Brownian Motion 
for i in range(1, N + 1):
    dZ = np.random.normal(size=M) * np.sqrt(dt) 
    # see eq 7 in longstaff-shcwartz 2001
    dS = TABLE[:, i - 1] * ((r * dt) + (sigma * dZ)) 
    TABLE[:, i] = TABLE[:, i -1] + dS

TABLE_DF = pd.DataFrame(TABLE)

mu = 0.06

# "brownian motion" graph 
TABLE_DF.transpose().plot(color="black", alpha=0.2)
plt.legend([])
plt.xlabel("time", fontdict=None, labelpad=None, loc=None)
plt.ylabel("price of underlying asset", fontdict=None, labelpad=None, loc=None)
plt.title(f'Puts with starting price: {PRICE} and strike price: {STRIKE_PRICE}')
plt.plot([0,N], [STRIKE_PRICE, STRIKE_PRICE], label="strike price")
plt.show()

# determining in-the-money 
Y = (STRIKE_PRICE - TABLE_DF[N]).map(lambda v: max(v, 0))
X = TABLE_DF[N-2]
IN_THE_MONEY = X < STRIKE_PRICE
X = X[IN_THE_MONEY]
Y = Y[IN_THE_MONEY]

# scatter plot
plt.scatter(X, Y)
plt.xlabel(f"price at {N-2}", fontdict=None, labelpad=None, loc=None)
plt.ylabel(f"amount profit at {N}", fontdict=None, labelpad=None, loc=None)
plt.title(f'Puts with starting price: {PRICE} and strike price: {STRIKE_PRICE}')
plt.show()

# set basis functions 
# TODO: SETTING BASIS FUNCTIONS 
poly = pd.DataFrame(index=X.index)
poly[0] = np.exp(-X/2)
poly[1] = np.exp(-X/2) * (1 - X)
poly[2] = np.exp(-X/2) * (1 - 2*X + X**2/2)


# stats models 
model = sm.OLS(Y, poly)
res = model.fit()
coef = res.params

# continuing / exercise 
continuation = (poly * coef).sum(axis=1)
exercise = (STRIKE_PRICE - TABLE_DF[N-2][IN_THE_MONEY])

# x scale 
x = np.linspace(.5, 1.5, 100)

# basis functions on y 
# TODO: SETTING BASIS FUNCTIONS 
y = np.exp(-x/2) * coef[0] + np.exp(-x/2) * (1 - x) * coef[1] + np.exp(-x/2) * (1 - 2*x + x**2/2) * coef[2]

# plotting 
plt.figure(figsize=(10,10))
plt.plot(x,y, linestyle=":", color="blue")
plt.xlabel(f"price at {N-2}", fontdict=None, labelpad=None, loc=None)
plt.ylabel(f"amount profit at {N}", fontdict=None, labelpad=None, loc=None)
plt.title(f'Puts with starting price: {PRICE} and strike price: {STRIKE_PRICE}')
plt.scatter(X, Y, marker="o",color="orange", label="Y (discounted exercise later, actual value that was gotten later)")
plt.scatter(X, continuation, label="continuation, theoretical", marker="x", color="blue")
plt.scatter(X, exercise, label="exercise now", marker="x", color="black")
plt.legend()
plt.show()
