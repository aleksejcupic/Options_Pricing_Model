import numpy
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import scipy

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


def f(X, n):
    return X ** n * numpy.exp(-X)

# df = pd.DataFrame(TABLE)
k = 1.1
# df.transpose().plot(color="red", alpha=0.3)
# plt.legend([])
# plt.plot([0,N], [k, k], label="strke price")
# plt.show()

# discounting the payoff
df = pd.DataFrame(TABLE)
for i in range(1):
    Y = pd.DataFrame()
    #(k - df[N]).map(lambda v: max(v, 0))
    for i in range(0, M):
        price = TABLE[i][N] - k
        if price > 0:
            Y.append(price)
        else:
            Y.append(0)
    # discount = numpy.exp(-mu * 1)
    # Y = Y * discount
    X = TABLE[N-1]
    ITM = X < k
    X = X[ITM]
    Y = Y[ITM]

    # laguerre polynomials 
    poly = pd.DataFrame(index=X.index)
    n = 10
    for i in range(0,n):
        poly[i] = numpy.exp(-X/2) * numpy.exp(X) / math.factorial(i) * scipy.derivative(f(X,i), X, dx=1e-6, n=i)
    # poly[0] = numpy.exp(-X/2)
    # poly[1] = numpy.exp(-X/2) * (1 - X)
    # poly[2] = numpy.exp(-X/2) * (1 - (2*X) + (X**2 / 2))

    # stats model 
    model = sm.OLS(Y, poly)
    res = model.fit()
    coef = res.params

    # determining whether continuing is better or not 
    continuation = (poly * coef).sum(axis=1)
    exercise = (k - TABLE[N-1][ITM])
    x = numpy.linspace(.5, 1.5, 100)
    y = 0
    for i in range(0, n):
        y += poly[i] * coef[i]
    continued = continuation > exercise
    continued = continued.reindex(TABLE.index).fillna(True)
    print(continued)

    plt.figure(figsize=(10,10))
    plt.plot(x,y, linestyle=":", color="blue")

    plt.scatter(X, Y, color="red", label="Y (discounted exercise later)")

    plt.scatter(X, continuation, label="continuation", marker="x", color="blue")
    plt.scatter(X, exercise, label="exercise now", marker="+", color="green")
    plt.legend()
