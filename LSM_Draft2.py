import numpy
import pandas
import matplotlib.pyplot
import statsmodels.api
import math
from scipy.misc import derivative

N = 1000 # number of time steps, number of discretization points, time points that the option can be exercised
M = 1000 # number of paths of the underlying asset
BF = 20 # number of basis functions used
T = 1 # time from 'now' to expiration of option in years 
dt = T/N # time between each time step 

PRICE = 1 # stock price 
STRIKE_PRICE = 1.1

# initialize table with stock price
TABLE = numpy.zeros((M, N + 1))
TABLE[:, 0] = PRICE

# Filling the table with stochastic diferential equation 
r = 0.06 # risk free rate (%)
sigma = 0.3 # constant for eq
# Z is standard Brownian Motion 
for i in range(1, N + 1):
    dZ = numpy.random.normal(size=M) * numpy.sqrt(dt) 
    # see eq 7 in longstaff-shcwartz 2001
    dS = TABLE[:, i - 1] * ((r * dt) + (sigma * dZ)) 
    TABLE[:, i] = TABLE[:, i -1] + dS

print(TABLE)
n = BF

# part of laguerre weighted polynomial function 
def f(X):
    return X ** n * numpy.exp(-X)

# creation of dataframe data type in order to 
# use with stats model functions 
TABLE_DF = pandas.DataFrame(TABLE)

# loop for backward induction on each time step from N to 0
# this is for put options
for i in range(1, N):
    print(i)
    # Y is the array of profit at time step N - i 
    Y = (STRIKE_PRICE - TABLE_DF[N - i]).map(lambda v: max(v, 0))
    # X is the price of each path at time N - i
    X = TABLE_DF[N - i]
    # IN_THE_MONEY is an array at time N - i that shows T/F
    # for whether the option is in the money
    IN_THE_MONEY = X < STRIKE_PRICE
    X = X[IN_THE_MONEY]
    Y = Y[IN_THE_MONEY]

    # laguerre polynomials 
    # poly is also a dataframe data type 
    poly = pandas.DataFrame(index=X.index)
    n = BF

    # see eq 5 in longstaff-schwartz 2001
    # polys are recreated at each iteration because indicies may change
    for i in range(0, n):
        n = i
        poly[i] = numpy.exp(-X/2) * numpy.exp(X) / math.factorial(i) * derivative(f, X, dx=1e-6, n=i, order=BF+1)

    # stats model 
    model = statsmodels.api.OLS(Y, poly)
    res = model.fit()
    coef = res.params

    # determining whether continuing is better or not 
    continuation = (poly * coef).sum(axis=1)
    exercise = (STRIKE_PRICE - TABLE_DF[N - i][IN_THE_MONEY])


    y = 0
    # see eq 6 in Longstaff, Schwartz 2001
    for p in range(0, n):
        y += poly[p] * coef[p]
    x = numpy.linspace(0.5, 1.5, len(y))
    continued = continuation > exercise
    continued = continued.reindex(TABLE_DF.index).fillna(True)
    print(continued)
    Y[continued] = 0
    print("x: ", x, "y: ", y)
    # plotting the graph
    # TABLE_DF.transpose().plot(color="red", alpha=0.3)

    matplotlib.pyplot.figure(figsize=(10,10))
    matplotlib.pyplot.plot(x,y, linestyle=":", color="blue")

    matplotlib.pyplot.scatter(X, Y, color="red", label="Y (discounted exercise later)")

    matplotlib.pyplot.scatter(X, continuation, label="continuation", marker="x", color="blue")
    matplotlib.pyplot.scatter(X, exercise, label="exercise now", marker="+", color="green")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()




# # loop for backward induction on each time step from N to 0
# # this is for call options
# for i in range(0, N - 1):
#     print(i)
#     # Y is the array of profit at time step N - i 
#     Y = (TABLE_DF[N - i] - STRIKE_PRICE).map(lambda v: max(v, 0))
#     # X is the price of each path at time N - i
#     X = TABLE_DF[N - i]
#     # IN_THE_MONEY is an array at time N - i that shows T/F
#     # for whether the option is in the money
#     IN_THE_MONEY = X > STRIKE_PRICE
#     X = X[IN_THE_MONEY]
#     Y = Y[IN_THE_MONEY]

#     # laguerre polynomials 
#     # poly is also a dataframe data type 
#     poly = pandas.DataFrame(index=X.index)
#     n = BF

#     # see eq 5 in longstaff-schwartz 2001
#     # polys are recreated at each iteration because indicies may change
#     for i in range(0, n):
#         n = i
#         poly[i] = numpy.exp(-X/2) * numpy.exp(X) / math.factorial(i) * derivative(f, X, dx=1e-6, n=i, order=15)

#     # stats model 
#     model = statsmodels.api.OLS(Y, poly)
#     res = model.fit()
#     coef = res.params

#     # determining whether continuing is better or not 
#     continuation = (poly * coef).sum(axis=1)
#     exercise = (TABLE_DF[N - i][IN_THE_MONEY] - STRIKE_PRICE)
#     x = numpy.linspace(.5, 1.5, 100)
#     y = 0
    
#     # see eq 6 in Longstaff, Schwartz 2001
#     for p in range(0, n):
#         y += poly[p] * coef[p]
#     continued = continuation < exercise
#     continued = continued.reindex(TABLE_DF.index).fillna(True)
#     print(continued)
#     Y[continued] = 0

#     # plotting the graphs
#     """matplotlib.figure(figsize=(10,10))
#     matplotlib.plot(x,y, linestyle=":", color="blue")

#     matplotlib.scatter(X, Y, color="red", label="Y (discounted exercise later)")

#     matplotlib.scatter(X, continuation, label="continuation", marker="x", color="blue")
#     matplotlib.scatter(X, exercise, label="exercise now", marker="+", color="green")
#     matplotlib.legend()"""