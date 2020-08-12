import numpy as np
import matplotlib.pyplot as plt

k = 0 # just auxiliary variable to number the figures

# read out data and save to arrays respectively
# with source
data = np.genfromtxt("cfd.dat", skip_header=3, delimiter=",").T
Lth = data[0]
Lcount = data[1]
Rth = data[2]
Rcount = data[3]

# figure 1
plt.figure(num=k, figsize=(10, 5))
plt.errorbar(Lth, Lcount, label="left", yerr=np.sqrt(Lcount))
plt.errorbar(Rth, Rcount, label="right", yerr=np.sqrt(Rcount))

plt.yscale('log')
plt.xlabel("threshold")
plt.ylabel("number of events in 4s")
plt.legend()
plt.savefig("../../report/figs/cfd.pdf")
plt.close(k)
k += 1

# without source
data = np.genfromtxt("cfd2.dat", skip_header=3, delimiter=",").T
Lth = data[0]
Lcount = data[1]
Rth = data[2]
Rcount = data[3]

# figure2
plt.figure(num=k, figsize=(10, 5))
plt.errorbar(Lth, Lcount, label="left", yerr=np.sqrt(Lcount))
plt.errorbar(Rth, Rcount, label="right", yerr=np.sqrt(Rcount))

plt.yscale('log')
plt.xlabel("threshold")
plt.ylabel("number of events in 4s")
plt.legend()
plt.savefig("../../report/figs/cfd2.pdf")
plt.close(k)
k += 1
