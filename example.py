import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

paths = [
    {0:1.00, 1:1.09, 2:1.08, 3:1.34},
    {0:1.00, 1:1.16, 2:1.26, 3:1.54},
    {0:1.00, 1:1.22, 2:1.07, 3:1.03},
    {0:1.00, 1:0.93, 2:0.97, 3:0.92},
    {0:1.00, 1:1.11, 2:1.56, 3:1.52},
    {0:1.00, 1:0.76, 2:0.77, 3:0.90},
    {0:1.00, 1:0.92, 2:0.84, 3:1.01},
    {0:1.00, 1:0.88, 2:1.22, 3:1.34},
]


df = pd.DataFrame(paths, index=range(1, len(paths)+1))
DISCOUNT = 0.06
STRIKE_PRICE = 1.10
N = 3
k = 1.1

df.transpose().plot()
plt.legend([])
plt.plot([0,N], [k, k], label="strike price", color="black", linestyle='--')

# Find options that are ITM.
Y = (k - df[N]).map(lambda v: max(v, 0)) 

Y = Y * np.exp(-DISCOUNT * 1)

X = df[N-1]

ITM = X < k

X = X[ITM]
Y = Y[ITM]


poly = pd.DataFrame(index=X.index)

poly[0] = 1
poly[1] = np.cos(X)
poly[2] = np.sin(X)
# poly[3] = np.cos(2*X)
# poly[4] = np.sin(2*X)

model = sm.OLS(Y, poly)
res = model.fit()
coef = res.params

continuation = (poly * coef).sum(axis=1)

exercise = (k - df[N-1][ITM])

x = np.linspace(.5, 1.5, 100)
y = 1 * coef[0] + np.cos(x) * coef[1] + np.sin(x) * coef[2]

plt.figure(figsize=(5,5))
plt.plot(x,y, linestyle=":", color="blue")

plt.scatter(X, Y, color="red", label="Y (discounted exercise later)")
plt.scatter(X, continuation, label="continuation", marker="x", color="blue")
plt.scatter(X, exercise, label="exercise now", marker="+", color="green")
plt.legend()

continued = continuation > exercise

# we also consider those that were OTM to be continued, since early exercise would have made no sense
# hence we can reindex and fill with True

continued = continued.reindex(df.index).fillna(True)
Y = (k - df[N-1]).map(lambda v: max(v, 0))
Y[continued] = 0  
Y = Y * np.exp(-DISCOUNT * 1)
X = df[N-2]
ITM = X < k
X = X[ITM]
Y = Y[ITM]
poly = pd.DataFrame(index=X.index)

poly[0] = np.exp(-X/2)
poly[1] = np.exp(-X/2) *(1-X)
poly[2] = np.exp(-X/2) * (1 - 2 * X + X ** 2 / 2)

# technically no need to recreate poly, however we do since the indices may change, or may increase or decrease

model = sm.OLS(Y, poly)
res = model.fit()
coef = res.params

continuation = (poly * coef).sum(axis=1)

exercise = (k - df[N-1][ITM])

x = np.linspace(.5, 1.5, 100)

y = np.exp(-x/2) * coef[0] + np.exp(-x/2) * (1-x) * coef[1] + np.exp(-x/2) * (1 - 2 * x + x ** 2 / 2) * coef[2]

plt.figure(figsize=(5,5))

plt.plot(x,y, linestyle=":", color="blue")
plt.scatter(X, Y, color="red", label="Y (discounted exercise later)")
plt.scatter(X, continuation, label="continuation", marker="x", color="blue")
plt.scatter(X, exercise, label="exercise now", marker="+", color="green")
plt.legend()
plt.show()