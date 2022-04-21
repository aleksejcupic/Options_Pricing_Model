import numpy as np 
import pylab as plt

# number of days to expiry
N_Days = 252

# number of Monte Carlo simulations to run
N_Runs = 100000

Spot_Price = 100
strike = 100

# annualised implied volatility
volatility = 0.15

np.random.seed(25)
rets = np.random.randn(N_Runs, N_Days)*volatility/np.sqrt(252)
rets.shape

traces = np.cumprod(1+rets,1)*Spot_Price

for i in traces[:100,:]:
    plt.plot(i)
plt.grid()
plt.xlabel('days', fontsize=12)
plt.ylabel('Spot price', fontsize=12)
plt.show()

plt.hist(traces[:,-1], bins=40);
plt.title('Distribution of final prices')
plt.xlabel('Final prices', fontsize=12)
plt.ylabel('counts')
plt.show()

call = np.mean((traces[:, -1] - strike)*((traces[:, -1] - strike) > 0))
call

put = np.mean((strike-traces[:,-1])*(((traces[:,-1]-strike)<0)))
put

def get_price_w_rf(right,T,S,X,v,rf,N=100000):
    D = np.exp(-rf*(T/252))
    prices = np.cumprod(1+(np.random.randn(T,N) * v / np.sqrt(252)),axis=0)*S
    if right=='c':
        return np.sum((prices[-1,:]-X*D)[prices[-1,:]>X*D])/prices.shape[1]
    else:
        return -np.sum((prices[-1,:]-X*D)[prices[-1,:]<X*D])/prices.shape[1]

get_price_w_rf('c',126,100,100,0.15,0.02)