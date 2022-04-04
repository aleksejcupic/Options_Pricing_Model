# Abstract: An implementation of the American style Option Pricing Model
# using the Least Squares Method for Monte Carlo Simulation from 
# Longstaff, Schwartz, Valuing American Options by Simulation (2001)

# The method is defined in their research. We implement it in Python

# By: Komi Alasse, Gio Canales, Aleksej Cupic 

import math
import sympy as sym
x = 0
n = 100

random_varible = 1 # X

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

original_derivative = (x ** n) * (math.e ** (-x))
nth_derivative(original_derivative, n)


def F(omega, t_k):
    # sum from j = 0 to inf of:
    #   a_j * laguerre_j(X)
    # as a_j coefficients are constants
    return 