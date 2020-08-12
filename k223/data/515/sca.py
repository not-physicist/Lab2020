import numpy as np
import matplotlib.pyplot as plt

k = 0
data = np.genfromtxt("sca-calibration.dat", skip_header=3, delimiter=",").T
channel = data[0]
Lcount = data[1]
Rcount = data[2]

plt.figure(num=k, figsize=(10, 5))
plt.errorbar(channel, Lcount, label="left", yerr=np.sqrt(Lcount))
plt.errorbar(channel, Rcount, label="right", yerr=np.sqrt(Rcount))

plt.xlabel("channels")
plt.ylabel("events in 10s")
plt.legend()
plt.savefig("../report/figs/sca.pdf")
plt.close(k)
k += 1

data = np.genfromtxt("sca-calibration2.dat", skip_header=3, delimiter=",").T
channel = data[0]
Lcount = data[1]
Rcount = data[2]

plt.figure(num=k, figsize=(10, 5))
plt.errorbar(channel, Lcount, label="left", yerr=np.sqrt(Lcount))
plt.errorbar(channel, Rcount, label="right", yerr=np.sqrt(Rcount))

plt.xlabel("channels")
plt.ylabel("events in 10s")
plt.legend()

plt.savefig("../report/figs/sca2.pdf")
plt.close(k)
k += 1
