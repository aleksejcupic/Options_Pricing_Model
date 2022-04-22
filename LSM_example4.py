def optionValue(S0, vol, T, K=40, M=50, I=4096, r=0.06):
    import numpy as np
    np.random.seed(150000)  # fix the seed for every valuation
    dt = T / M  # time interval
    df = np.exp(-r * dt)  # discount factor per time time interval
    # Simulation of Index Levels
    S = np.zeros((M + 1, I), 'd')  # stock price matrix
    S[0, :] = S0  # intial values for stock price
    for t in range(1, M + 1):
        ran = np.random.standard_normal(I / 2)
        ran = np.concatenate((ran, -ran))  # antithetic variates
        ran = ran - np.mean(ran)  # correct first moment
        ran = ran / np.std(ran)  # correct second moment
        S[t, :] = S[t - 1, :] * np.exp((r - vol ** 2 / 2) * dt
                        + vol * ran * np.sqrt(dt))
    h = np.maximum(K - S, 0)  # inner values for put option
    V = np.zeros_like(h)  # value matrix
    V[-1] = h[-1]
    # Valuation by LSM
    for t in range(M - 1, 0, -1):
        rg = np.polyfit(S[t, :], V[t + 1, :] * df, 5)  # regression
        C = np.polyval(rg, S[t, :])  # evaluation of regression
        V[t, :] = np.where(h[t, :] > C, h[t, :],
                         V[t + 1, :] * df)  # exercise decision/optimization
    V0 = np.sum(V[1, :] * df) / I  # LSM estimator
    print("S0 %4.1f|vol %4.2f|T %2.1f| Option Value %8.3f" % (S0, vol, T, V0))
    return V0

def seqValue():
    optionValues = []
    for S0 in (36., 38., 40., 42., 44.):  # initial stock price values
        for vol in (0.2, 0.4):  # volatility values
            for T in (1.0, 2.0):  # times-to-maturity
                optionValues.append(optionValue(S0, vol, T))
    return optionValues

from time import time
t0 = time()
optionValues = seqValue()  # calculate all values
t1 = time(); d1 = t1 - t0
print("Duration in Seconds %6.3f" % d1)

# in the shell ...
####
# ipcluster start -n 4
####
# or maybe more cores


from IPython.parallel import Client
cluster_profile = "default"
# the profile is set to the default one
# see IPython docs for further details
c = Client(profile=cluster_profile)
view = c.load_balanced_view()

def parValue():
    optionValues = []
    for S in (36., 38., 40., 42., 44.):
        for vol in (0.2, 0.4):
            for T in (1.0, 2.0):
                value = view.apply_async(optionValue, S, vol, T)
                # asynchronously calculate the option values
                optionValues.append(value)
    return optionValues

def execution():
    optionValues = parValue()  # calculate all values
    print("Submitted tasks %d" % len(optionValues))
    c.wait(optionValues)
    # wait for all tasks to be finished
    return optionValues
t0 = time()
optionValues = execution()
t1 = time(); d2 = t1 - t0
print("Duration in Seconds %6.3f" % d2)

d1 / d2  # speed-up of parallel execution

optionValues[0].result

optionValues[0].metadata

for result in optionValues:
    print(result.metadata['stdout'],)