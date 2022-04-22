import numpy as np
import datetime
# initial derivative parameters 
S = 101.15          #stock price
K = 98.01           #strike price
vol = 0.0991        #volatility (%)
r = 0.01            #risk-free rate (%)
N = 100              #number of time steps
M = 100000           #number of simulations

market_value = 3.86 #market price of option
T = ((datetime.date(2022,7,17)-datetime.date.today()).days+1)/365    #time in years
print(T)

#precompute constants
dt = T/N
print("dt ", dt)
nudt = (r - 0.5*vol**2)*dt
print("nudt: ", nudt)
volsdt = vol*np.sqrt(dt)
print("volsdt: ", volsdt)
lnS = np.log(S)
print("lnS: ", lnS)


# Monte Carlo Method
print("Monte Carlo Method")
Z = np.random.normal(size=(N, M)) 
print("Z: ", Z)
delta_lnSt = nudt + volsdt*Z 
print("delta_lnSt: ", delta_lnSt)
lnSt = lnS + np.cumsum(delta_lnSt, axis=0)
print("lnSt: ", lnSt)
lnSt = np.concatenate( (np.full(shape=(1, M), fill_value=lnS), lnSt ) )
print("lnSt: ", lnSt)

# Compute Expectation and SE
print("Compute Expectation and SE")
ST = np.exp(lnSt)
print("ST: ", ST)
CT = np.maximum(0, ST - K)
print("CT: ", CT)
C0 = np.exp(-r*T)*np.sum(CT[-1])/M
print("C0: ", C0)

sigma = np.sqrt( np.sum( (CT[-1] - C0)**2) / (M-1) )
print("sigma: ", sigma)
SE = sigma/np.sqrt(M)
print("SE: ", SE)

print("Call value is ${0} with SE +/- {1}".format(np.round(C0,6),np.round(SE,6)))

