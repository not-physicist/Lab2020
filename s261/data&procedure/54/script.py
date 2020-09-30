import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.genfromtxt("./timedel_sorted.dat", delimiter='    ').T
#  print(data)

date = data[0]
magA = data[1] + 1.8
sigmaMagA = data[2]
magB = data[3]
sigmaMagB = data[4]

k = 0
plt.figure(num=k, figsize=(10,7))
plt.errorbar(date, magA, yerr=sigmaMagA, label="Image A (mag+1.8)", capsize=2, fmt="o-", color="black")
plt.errorbar(date, magB, yerr=sigmaMagB, label="Image B", capsize=2, fmt="^-", color="black")

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xticks(np.linspace(np.amin(date), np.amax(date), 12))
#  plt.xlim(np.amin(date), np.amax(date))
plt.legend()
plt.xlabel("Julian date")
plt.ylabel("Magnitude")
plt.savefig("timeDelay.pdf", bbox_inches="tight")
plt.close(k)
k += 1


data = np.genfromtxt("./disp.out_MCprob.tab", delimiter="\t").T
timeDel = data[0]
prob = data[1]

def gauss(x, mu, sigma):
    return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/sigma**2/2)

plt.figure(num=k, figsize=(10,7))
plt.scatter(timeDel, prob, color="black", marker="^", label="from MC")

popt, pcov = curve_fit(gauss, timeDel, prob, p0=[34, 2])
print("Parameters", popt)
print("Covariance matrix", pcov)
xArray = np.linspace(np.amin(timeDel), np.amax(timeDel), 100)
probFit = gauss(xArray, *popt)
plt.plot(xArray, probFit, label="fit", color="black")
plt.legend()

plt.xlabel("Time delay")
plt.ylabel("Probability")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("MCProb.pdf", bbox_inches="tight")
plt.close(k)
k += 1
