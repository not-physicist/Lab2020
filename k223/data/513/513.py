import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit

k = 0
data = np.genfromtxt("513.dat", skip_header=3, delimiter=",").T
delay = data[0]
countRate = data[1] / 4
sigmaCountRate = np.sqrt(data[1])/4

plt.errorbar(delay, countRate, fmt="o", label="data",
             yerr = sigmaCountRate, ms=3, capsize=2, color="black")

# in the end, 42ns delay

def FitFunc(t, t0, t1, A0, A1, sigma):
    return (A0 + A1*(1/2 + erf((t-t0)/sigma) * erf((t1-t)/sigma) ) )

initParaGuess = [25, 55, 300, 300, 10]

#  plt.plot(delay, FitFunc(delay, *initParaGuess))

popt, pcov = curve_fit(FitFunc, delay, countRate, p0=initParaGuess)
print("Parameters are:", popt)
print("Covariace matrix is:", np.sqrt(np.diag(pcov)))

tArray = np.linspace(np.amin(delay), np.amax(delay), 500)
plt.plot(tArray, FitFunc(tArray, *popt), label="fit")


plt.xlabel("delay[ns]")
plt.ylabel("event rate[s]")
plt.legend(loc=2)
plt.savefig("../../report/figs/prompt.pdf", bbox_inches="tight")
plt.close()

