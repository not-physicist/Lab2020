import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def Line(x, a, b):
    return a*x + b


data = np.genfromtxt("bot.dat", delimiter=",").T

voltage = data[0]
counts = data[1]
sigma = np.sqrt(data[1])

avrg_voltage = np.zeros(int(voltage.shape[0]/4))
avrg_counts = np.zeros(int(voltage.shape[0]/4))
avrg_sigma = np.zeros(int(voltage.shape[0]/4))

for i in range(0, voltage.shape[0], 4):
    #  print(i)
    #  print(voltage[i:i+4])
    #  print(counts[i:i+4])
    if voltage[i] == voltage[i+3]:
        avrg_voltage[int(i/4)] = voltage[i]
        avrg_counts[int(i/4)] = np.mean(counts[i:i+4])
        avrg_sigma[int(i/4)] = 1/4 * np.sqrt(np.sum(sigma[i:i+4]**2))


k = 0
plt.figure(num=k, figsize=(10,7))
plt.errorbar(avrg_voltage, avrg_counts, yerr=avrg_voltage, fmt="o", label="data", color="black")

mask = (avrg_voltage > 2099)
mask_voltage = avrg_voltage[mask]
mask_counts = avrg_counts[mask]
mask_sigma = avrg_sigma[mask]
#  print(mask_voltage, mask_counts, mask_sigma)
popt, pcov = curve_fit(Line, mask_voltage, mask_counts, absolute_sigma=True, sigma=mask_sigma)
#  print(popt, pcov)

x = np.linspace(2000, 2300, 200)
plt.plot(x, Line(x, *popt), label="fit", color="black")

plt.xlabel("voltage[V]")
plt.ylabel("Counts in 10s")
plt.legend(loc=2)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig("bot.pdf", bbox_inches="tight")
plt.close(k)
k += 1
#############################################################
# top
#############################################################

data = np.genfromtxt("top.dat", delimiter=",").T

voltage = data[0]
counts = data[1]
sigma = np.sqrt(data[1])

avrg_voltage = np.zeros(int(voltage.shape[0]/4))
avrg_counts = np.zeros(int(voltage.shape[0]/4))
avrg_sigma = np.zeros(int(voltage.shape[0]/4))

for i in range(0, voltage.shape[0], 4):
    #  print(i)
    #  print(voltage[i:i+4])
    #  print(counts[i:i+4])
    if voltage[i] == voltage[i+3]:
        avrg_voltage[int(i/4)] = voltage[i]
        avrg_counts[int(i/4)] = np.mean(counts[i:i+4])
        avrg_sigma[int(i/4)] = 1/4 * np.sqrt(np.sum(sigma[i:i+4]**2))

plt.figure(num=k, figsize=(10,7))
plt.errorbar(avrg_voltage, avrg_counts, yerr=avrg_sigma, fmt="o", label="data", color="black")

plt.xlabel("voltage[V]")
plt.ylabel("Coincidences in 10s")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig("top.pdf", bbox_inches="tight")
plt.close(k)
k += 1
