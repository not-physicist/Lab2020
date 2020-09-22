import numpy as np
import matplotlib.pyplot as plt
import os

def chi2(exp, obs):
    return np.sum((exp - obs)**2 / exp)

def Sort2Array(x, y):
    temp = np.array([x,y]).T
    #  print(temp)
    temp = temp[temp[:,0].argsort()].T
    return temp[0], temp[1]

k = 0
directory = "./"
chi2Array = np.array([])
timedelArray = np.array([])
for filename in os.listdir(directory):
    if filename.startswith('disp.') and filename.endswith('.dat'):
        #  print(filename)
        timedelBin = filename.replace('disp.out_magshiftbin_', '').replace('.dat', '')
        timedelBin = int(timedelBin)
        timedelArray = np.append(timedelArray, timedelBin*0.1 - 2.5)
        data = np.genfromtxt(filename, delimiter=" ").T
        #  print(data)
        timedel = data[0]
        measDisp = data[1]
        measFit = data[2]
        chi2Array = np.append(chi2Array, chi2(measFit, measDisp))

        plt.figure(num=k, figsize=(10,6))
        plt.plot(timedel, measDisp, label="measured", marker="o", color="black")
        plt.plot(timedel, measFit, label="fitted", color="black")
        plt.legend()
        plt.xlabel("Time delay [days]")
        plt.ylabel("Dispersion")

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.savefig(filename + ".pdf", bbox_inches="tight")
        plt.close(k)
        k+=1

timedelArray, chi2Array = Sort2Array(timedelArray, chi2Array)
#  print(chi2Array)
#  print(timedelArray)
plt.figure(num=k, figsize=(10,6))
plt.plot(timedelArray, chi2Array, marker="o")
plt.xlabel("Magnitude difference")
plt.ylabel("$\chi^2$")
plt.savefig("chi2.pdf", bbox_inches="tight")
