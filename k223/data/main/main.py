import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("main.dat", skip_header=3, delimiter=",").T
angle = data[0]
duration = data[1]
Lcounter = data[2]
coinCounter = data[3]
Rcounter = data[4]
sigmaTheta = 2 # degree

coincRate = coinCounter/duration
sigmaCoincRate = np.sqrt(coinCounter)/duration

k = 0
plt.figure(num=k, figsize=(10,5))
plt.errorbar(angle, coincRate, label="coincidence",
             yerr=sigmaCoincRate, xerr=sigmaTheta, fmt="o")

plt.ylabel("coincidence rate[$s^{-1}$]")
plt.xlabel("angle")
plt.savefig("../report/figs/angCor.pdf", bbox_inches="tight")

plt.close(k)
k += 1

LcountRate = Lcounter/duration
sigmaLcountRate = np.sqrt(Lcounter)/duration
RcountRate = Rcounter/duration
sigmaRcountRate = np.sqrt(Rcounter)/duration

k = 0
plt.figure(num=k, figsize=(10,5))

plt.errorbar(angle, LcountRate, yerr=sigmaLcountRate, xerr=sigmaTheta, fmt="o")
plt.errorbar(angle, RcountRate, yerr=sigmaRcountRate, xerr=sigmaTheta, fmt="o")

plt.ylabel("count rate[$s^{-1}$]")
plt.xlabel("angle")
plt.savefig("../report/figs/countRate.pdf", bbox_inches="tight")

plt.close(k)
k += 1
