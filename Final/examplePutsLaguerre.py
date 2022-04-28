import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# EXAMPLE PUTS USING LAGUERRE POLYNOMIALS 
PRICE = 1
STRIKE_PRICE = 1
N = 3 # number of time steps 

paths = [
    {0:1.00, 1:1.09, 2:1.08, 3:1.34},
    {0:1.00, 1:1.16, 2:1.26, 3:1.54},
    {0:1.00, 1:1.22, 2:1.07, 3:1.03},
    {0:1.00, 1:0.93, 2:0.97, 3:0.92},
    {0:1.00, 1:1.11, 2:1.56, 3:1.52},
    {0:1.00, 1:0.76, 2:0.77, 3:0.90},
    {0:1.00, 1:0.92, 2:0.84, 3:1.01},
    {0:1.00, 1:0.88, 2:1.22, 3:1.34},
]

TABLE = pd.DataFrame(paths, index=range(1, len(paths)+1))

mu = 0.06

# "brownian motion" graph 
TABLE.transpose().plot(color="red", alpha=0.3)
plt.legend([])
plt.plot([0,N], [STRIKE_PRICE, STRIKE_PRICE], label="strike price")
plt.show()

# determining in-the-money 
Y = (STRIKE_PRICE - TABLE[N]).map(lambda v: max(v, 0))
X = TABLE[N-1]
IN_THE_MONEY = X < STRIKE_PRICE
X = X[IN_THE_MONEY]
Y = Y[IN_THE_MONEY]

# scatter plot
plt.scatter(X, Y)
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
exercise = (STRIKE_PRICE - TABLE[N-1][IN_THE_MONEY])

# x scale 
x = np.linspace(.5, 1.5, 100)

# basis functions on y 
# TODO: SETTING BASIS FUNCTIONS 
y = np.exp(-x/2) * coef[0] + np.exp(-x/2) * (1 - x) * coef[1] + np.exp(-x/2) * (1 - 2*x + x**2/2) * coef[2]

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
