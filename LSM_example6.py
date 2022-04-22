    # """ AMERICAN OPTION PRICING BY LEAST SQUARES MONTE CARLO, FINITE DIFFERENCE, ANALYTICAL AND BINOMIAL METHODS """
# need arm64, not x86
from numpy import zeros, concatenate, sqrt, exp, maximum, polyfit, polyval, shape, where, sum, argsort, random, \
    RankWarning, put, nonzero
from zlib import compress
import matplotlib.pyplot as plt
import os
import sys
from QuantLib import *
from pylab import *
import warnings
warnings.simplefilter('ignore', RankWarning)

plt.style.use('seaborn')

# Define global parameters
S0 = 100                                                       # Underlying price
K = 90                                                         # Strike
valuation_date = Date(1, 1, 2018)                              # Valuation date
expiry_date = Date(1, 1, 2019)                                 # Expiry date
t = ActualActual().yearFraction(valuation_date, expiry_date)   # Year fraction
T = 100                                                        # Time grid
dt = t / T                                                     # Delta time
r = 0.01                                                       # Interest rate
sig = 0.4                                                      # Volatility
sim = 10 ** 5                                                  # Number of MC simulations
DiscountFactor = exp(-r * dt)                                  # Discount factor

""" Least Squares Monte Carlo """


def GBM(underlying, time, simulations, rate, sigma, delta_t):  # Geometric Brownian Motion
    GBM = zeros((time + 1, simulations))
    GBM[0, :] = underlying
    for t in range(1, time + 1):
        brownian = standard_normal(simulations // 2)
        brownian = concatenate((brownian, -brownian))
        GBM[t, :] = (GBM[t - 1, :] * exp((rate - sigma ** 2 / 2.) * delta_t + sigma * brownian * sqrt(delta_t)))
    return GBM


def Payoff(strike, paths, simulations):  # Define option type and respective payoff
    if OptionType == 'call':
        po = maximum(paths - strike, zeros((T + 1, simulations)))
    elif OptionType == 'put':
        po = maximum(strike - paths, zeros((T + 1, simulations)))
    else:
        print('Incorrect input')
        os.execl(sys.executable, sys.executable, *sys.argv)
    return po


def loadingBar(count, total, size):  # MC progress bar
    percent = float(count) / float(total) * 100
    sys.stdout.write("\r" + str(int(count)).rjust(3, '0') + "/" + str(int(total)).rjust(3, '0') + ' [' + '=' * int(
        percent / 10) * size + ' ' * (10 - int(percent / 10)) * size + ']')


# Graph the regression fit and simulations
OptionType = str(input('Price call or put:'))
print('Plotting fitted regression at T...')
GBM = GBM(S0, T, sim, r, sig, dt)
payoff = Payoff(K, GBM, sim)
ValueMatrix = zeros_like(payoff)
ValueMatrix[T, :] = payoff[T, :]
prices = GBM[T, :]
value = ValueMatrix[T, :]
regression = polyfit(prices, value * DiscountFactor, 4)
ContinuationValue = polyval(regression, prices)
sorted_index = argsort(prices)
prices = prices[sorted_index]
ContinuationValue = ContinuationValue[sorted_index]

ValueMatrix[T, :] = where(payoff[T, :] > ContinuationValue, payoff[T, :], ValueMatrix[T, :] * DiscountFactor)
ValueVector = ValueMatrix[T, :] * DiscountFactor
ValueVector = ValueVector[sorted_index]

plt.figure()
f, axes = plt.subplots(2, 1)
axes[0].set_title('American Option')
axes[0].plot(prices, ContinuationValue, label='Fitted Polynomial')
axes[0].plot(prices, ValueVector, label='Inner Value')
axes[0].set_ylabel('Payoff')
axes[0].set_xlabel('Asset Price')
axes[0].legend()
axes[1].set_title('Geometric Brownian Motion')
axes[1].plot(GBM, lw=0.5)
axes[1].set_ylabel('Asset Price')
axes[1].set_xlabel('Time')
f.tight_layout()
plt.show()

# MC results
print('Pricing option...')
for i in range(0, 100):
    loadingBar(i, 100, 2)
    for t in range(T - 1, 0, -1):
        ITM = payoff[t, :] > 0
        ITMS = compress(ITM, GBM[t, :])
        ITMP = compress(ITM, payoff[t + 1, :] * DiscountFactor)
        regression = polyval(polyfit(ITMS, ITMP, 4), ITMS)
        continuation = zeros(sim)
        put(continuation, nonzero(ITM), regression)
        payoff[t, :] = where(payoff[t, :] > continuation, payoff[t, :], payoff[t + 1, :] * DiscountFactor)
        price = sum(payoff[1, :] * DiscountFactor) / sim
print('\nLeast Squares Monte Carlo Price:', price)


""" QuantLib Pricing """

S0 = SimpleQuote(S0)
if OptionType == 'call':
    OptionType = Option.Call
elif OptionType == 'put':
    OptionType = Option.Put
else:
    print('Incorrect input')
    os.execl(sys.executable, sys.executable, *sys.argv)


def Process(valuation_date, r, dividend_rate, sigma, underlying):
    calendar = UnitedStates()
    day_counter = ActualActual()
    Settings.instance().evaluation_date = valuation_date
    interest_curve = FlatForward(valuation_date, r, day_counter)
    dividend_curve = FlatForward(valuation_date, dividend_rate, day_counter)
    volatility_curve = BlackConstantVol(valuation_date, calendar, sigma, day_counter)
    u = QuoteHandle(underlying)
    d = YieldTermStructureHandle(dividend_curve)
    r = YieldTermStructureHandle(interest_curve)
    v = BlackVolTermStructureHandle(volatility_curve)
    return BlackScholesMertonProcess(u, d, r, v)


def FDAmericanOption(valuation_date, expiry_date, OptionType, K, process):  # Finite difference
    exercise = AmericanExercise(valuation_date, expiry_date)
    payoff = PlainVanillaPayoff(OptionType, K)
    option = VanillaOption(payoff, exercise)
    time_steps = 100
    grid_points = 100
    engine = FDAmericanEngine(process, time_steps, grid_points)
    option.setPricingEngine(engine)
    return option


def ANAmericanOption(valuation_date, expiry_date, OptionType, K, process):  # Analytical
    exercise = AmericanExercise(valuation_date, expiry_date)
    payoff = PlainVanillaPayoff(OptionType, K)
    option = VanillaOption(payoff, exercise)
    engine = BaroneAdesiWhaleyEngine(process)
    option.setPricingEngine(engine)
    return option


def BINAmericanOption(valuation_date, expiry_date, OptionType, K, process):  # Binomial
    exercise = AmericanExercise(valuation_date, expiry_date)
    payoff = PlainVanillaPayoff(OptionType, K)
    option = VanillaOption(payoff, exercise)
    timeSteps = 10 ** 3
    engine = BinomialVanillaEngine(process, 'crr', timeSteps)
    option.setPricingEngine(engine)
    return option


def FDAmericanResults(option):
    print('Finite Difference Price: ', option.NPV())
    print('Option Delta: ', option.delta())
    print('Option Gamma: ', option.gamma())


def ANAmericanResults(option):
    print('Barone-Adesi-Whaley Analytical Price: ', option.NPV())


def BINAmericanResults(option):
    print('Binomial CRR Price: ', option.NPV())


# Quantlib results
process = Process(valuation_date, r, 0, sig, S0)
ANoption = ANAmericanOption(valuation_date, expiry_date, OptionType, K, process)
ANAmericanResults(ANoption)
BINoption = BINAmericanOption(valuation_date, expiry_date, OptionType, K, process)
BINAmericanResults(BINoption)
FDoption = FDAmericanOption(valuation_date, expiry_date, OptionType, K, process)
FDAmericanResults(FDoption)

os.system('say "completo"')