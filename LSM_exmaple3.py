import ​math
import ​random
import ​pylab
from ​random ​import ​gauss from ​math ​import ​exp, sqrt import ​matplotlib
import ​matplotlib.pyplot ​as ​plt from ​numpy ​import ​*
import ​numpy ​as ​np
# Monte carlo simulation using brownian motion
# n is the number of simulations, T is time to maturity,
# sigma is the volatility , S0 is the initial stock price, K is the strike price.
def ​monte_carlo_simulation(n,T,sigma,S0,​K​): dt=T*​252
​# generate random numbers
u = random.randn(dt) * sigma / sqrt(dt)
​#lets plot the histogram plt.hist(u)
plt.title(​"Gaussian Histogram"​) plt.xlabel(​"Value"​) plt.ylabel(​"Frequency"​) plt.show()
​# creating a geometric random walk
z= cumprod(​1​+random.randn(n,dt)*sigma/sqrt(dt),​1​)*S0 plt.plot(z)
plt.show()
plt.title(​"Geometric Brownian Motion"​) plt.xlabel(​"time t"​)
plt.ylabel(​"Stock Price"​)
​for ​i ​in ​z: plt.plot(i) plt.show() plt.hist(z[:,-​1​],​40​) plt.show()
​# to compute the payoff of the option payoffs = (z[:, -​1​] - ​100​) * ((z[:, -​1​] - ​100​) > ​0​) ​#print(payoffs)
price = mean(payoffs) ​print​(price)
# calling the function monte_carlo_simulation(n,T,sigma,S0,K)
monte_carlo_simulation(​1000​,​2​,​0.3​,​100​,​105​)