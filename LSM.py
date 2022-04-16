# Abstract: An implementation of the American style Option Pricing Model
# using the Least Squares Method for Monte Carlo Simulation from 
# Longstaff, Schwartz, Valuing American Options by Simulation (2001)

# The method is defined in their research. We implement it in Python

# By: Komi Alasse, Gio Canales, Aleksej Cupic 

# IMPORTS
import math
import sympy as sym

# GLOBAL VARIABLES FOR TESTING:
x = 0
k = 10 # number of discretization points
    # time points that the option can be exercised
b = 10 # number of basis functions used
p = 3 # number of paths that the price of the
    # underlying asset can take from each t_k to t_k+1
n = 100 

random_varible = 1 # X


# FUNCTIONS
def laguerre(X, n):
    Ln_of_X = 0
    part_1 = math.e ** (-X / 2)
    part_2 = (math.e ** X) / math.factorial(n)
    #derivative = (X ** n) * (math.e ** (-X))
    #part_3 = nth_derivative(derivative, n)


def nth_derivative(diff, n):
    new_diff = 0
    sym.diff(diff)
    nth_derivative(new_diff, n - 1)

def chebyshev_polynomial(X, n):
    return

def chebyshev_polynomial_sums(X,n):
    result = 0
    for m in range(0, int(n / 2)):
        result += math.comb(n, 2*m) * (X ** (n - (2*m))) * ((x ** 2 - 1) ** m)
    return result

def F_laguerre(omega, t_k):
    # sum from j = 0 to inf of:
    #   a_j * laguerre_j(X)
    # as a_j coefficients are constants
    return 

def F_chebyshev_sum(omega, t_k):
    # using the first M basis functions
    # sum from j = 0 to inf of:
    #   a_j * chebyshev_polynomials_sums_j(X)
    result = 0
    n = 10
    X = 1
    a = 1 # as a_j coefficients are constants
    for j in range(0, n):
        result += a * chebyshev_polynomial_sums(X, n) 



# MAIN - TESTING 
print(F_chebyshev_sum(1, 1))

# original_derivative = (x ** n) * (math.e ** (-x))
# nth_derivative(original_derivative, n)