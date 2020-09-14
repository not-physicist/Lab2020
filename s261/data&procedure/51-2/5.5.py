import numpy as np
import matplotlib.pyplot as plt

def Sort2Array(x, y):
    temp = np.array([x,y]).T
    print(temp)
    temp = temp[temp[:,0].argsort()].T
    return temp[0], temp[1]

exptime = np.genfromtxt("terminal.out2", delimiter="\t", skip_header=1).T[1]
#  print(exptime)
mean = np.genfromtxt("terminal.out", delimiter="\t").T[5]
sigma = np.genfromtxt("terminal.out", delimiter="\t").T[6]
#  print(mean)
exptime, mean = Sort2Array(exptime, mean)

plt.figure(figsize=[8,5])

color = "tab:blue"
plt.errorbar(exptime, mean, yerr=sigma, fmt="o-", color=color, capsize=2)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel("Exposture time [s]")
plt.ylabel("Mean value", color=color)
ax.tick_params(axis="y", labelcolor=color)

color="tab:red"
ratio = mean/exptime
#  print(ratio)
ax2 = ax.twinx()
ax2.errorbar(exptime, ratio, yerr=sigma/exptime, color=color, capsize=2, fmt="o-")
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_ylabel("ratio of mean value and exptime", color=color)

plt.savefig("./mean_exp.pdf", bbox_inches="tight")
plt.close()

