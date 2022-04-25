# only works on arm64, not x86
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

T = 1
N = 3
dt = T/N
M = 200
S = np.zeros(shape=(M, N+1))
S[:, 0] = 1 # starting price of the underlying asset

k = 1.1
mu = 0.06
si = 0.3

for i in range(1, N+1):
  dz = np.random.normal(size=M) * np.sqrt(dt)
  dS = ((mu * dt) + (si * dz)) * S[:, i-1]
  S[:, i] = S[:, i-1] + dS


print(S)

df = pd.DataFrame(S)

# df.transpose().plot(color="red", alpha=0.3)
# plt.legend([])
# plt.plot([0,N], [k, k], label="strke price")
# plt.show()

df

mu = 0.06
k = 1.10

Y = (k - df[N]).map(lambda v: max(v, 0))  # put payoff 

# discount the payoff
discount = np.exp(-mu * 1)
Y = Y * discount
# Y
X = df[N-1]
ITM = X < k
X = X[ITM]
# X
Y = Y[ITM]
# Y
# plt.scatter(X, Y)

poly = pd.DataFrame(index=X.index)
poly[0] = np.exp(-X / 2)
poly[1] = np.exp(-X / 2) * (1 - X) 
poly[2] = np.exp(-X / 2) * (1 - 2 * X + X ** 2 / 2) 

# poly[3] = np.cos(2*X)
# poly[4] = np.sin(2*X)

poly

model = sm.OLS(Y, poly)
res = model.fit()
coef = res.params
print("----")
print(coef, res)
print("----")


continuation = (poly * coef).sum(axis=1)

exercise = (k - df[N-1][ITM])


x = np.linspace(.5, 1.5, 100)
# y = 1 * coef["x0"] + x * coef["x"] + x**2 * coef["x2"]# + x**3 * coef["x3"]

y = np.exp(-X / 2) * coef[0] + np.exp(-X / 2) * (1 - X)  * coef[1] + np.exp(-X / 2) * (1 - 2 * X + X ** 2 / 2)  * coef[2]# + np.cos(2*x) * coef[3] + np.sin(2*x) * coef[4]

plt.figure(figsize=(10,10))
plt.plot(x,y, linestyle=":", color="blue")

plt.scatter(X, Y, color="red", label="Y (discounted exercise later)")

plt.scatter(X, continuation, label="continuation", marker="x", color="blue")
plt.scatter(X, exercise, label="exercise now", marker="+", color="green")
plt.legend()
plt.show()


continued = continuation > exercise

# we also consider those that were OTM to be continued, since early exercise would have made no sense
# hence we can reindex and fill with True

continued = continued.reindex(df.index).fillna(True)

print(continued)

# Y = (k - df[N-1]).map(lambda v: max(v, 0))

# Y[continued] = 0  # we take realised cash flows, and if we continue no cash is realised

# discount = np.exp(-mu * 1) # discount again 1 since we realised those next iteration

# Y = Y * discount
# Y

# X = df[N-2]

# ITM = X < k

# X = X[ITM]

# X

# Y = Y[ITM]
# Y

# poly = pd.DataFrame(index=X.index)

# poly[0] = 1
# poly[1] = np.cos(X)
# poly[2] = np.sin(X)

# # technically no need to recreate poly, however we do since the indices may change, or may increase or decrease

# model = sm.OLS(Y, poly)
# res = model.fit()
# coef = res.params

# continuation = (poly * coef).sum(axis=1)

# exercise = (k - df[N-1][ITM])

# x = np.linspace(.5, 1.5, 100)

# y = 1 * coef[0] + np.cos(x) * coef[1] + np.sin(x) * coef[2]

# plt.figure(figsize=(10,10))

# plt.plot(x,y, linestyle=":", color="blue")

# plt.scatter(X, Y, color="red", label="Y (discounted exercise later)")

# plt.scatter(X, continuation, label="continuation", marker="x", color="blue")
# plt.scatter(X, exercise, label="exercise now", marker="+", color="green")
# plt.legend()
