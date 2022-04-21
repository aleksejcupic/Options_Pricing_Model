# valuing an American option

from QuantLib import *

valuation_date = Date(22, 8, 2018)
Settings.instance().evaluationDate = valuation_date+2

calendar = Canada()
volatility = 42.66/100
day_count = Actual365Fixed()

underlying = 13.5
risk_free_rate = 2.13/100
dividend_rate = 1.2/100

exercise_date = Date(22, 8, 2021)
strike = 13
option_type = Option.Put

payoff = PlainVanillaPayoff(option_type, strike)
exercise = EuropeanExercise(exercise_date)
european_option = VanillaOption(payoff, exercise)

spot_handle = QuoteHandle(SimpleQuote(underlying))

flat_ts = YieldTermStructureHandle(FlatForward(valuation_date,risk_free_rate,day_count))
dividend_yield = YieldTermStructureHandle(FlatForward(valuation_date,dividend_rate,day_count))
flat_vol_ts = BlackVolTermStructureHandle(BlackConstantVol(valuation_date,calendar,volatility,day_count))
bsm_process = BlackScholesMertonProcess(spot_handle,dividend_yield,flat_ts,flat_vol_ts)

# European option
european_option.setPricingEngine(AnalyticEuropeanEngine(bsm_process))
bs_price = european_option.NPV()
print("European option price is ", bs_price)


# American option MC
MC_engine = MCAmericanEngine(bsm_process, 'PseudoRandom', timeSteps=50, polynomOrder=5, seedCalibration=42, requiredSamples=10 ** 5)
am_exercise = AmericanExercise(valuation_date, exercise_date)
MCamerican_option = VanillaOption(payoff, am_exercise)
MCamerican_option.setPricingEngine(MC_engine)
MC_price = MCamerican_option.NPV()
print("MC American option price is ", MC_price)