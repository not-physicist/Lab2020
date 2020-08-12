import numpy as np
import matplotlib.pyplot as plt

k = 0
data = np.genfromtxt("513.dat", skip_header=3, delimiter=",").T
delay = data[0]
count = data[1]
#TODO: add error, sqrt(N)

plt.plot(delay, count)
plt.xlabel("delay[ns]")
plt.ylabel("number of events in 4s")
plt.savefig("../../report/figs/prompt.pdf", bbox_inches="tight")
plt.close()
# in the end, 42ns delay
