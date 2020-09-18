import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("./timedel/timedel_sorted.dat", delimiter='    ').T
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
