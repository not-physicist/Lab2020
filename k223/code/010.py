import numpy as np
import matplotlib.pyplot as plt

def W(theta):
    return 1 + 1*np.cos(theta)**2

thetaArray = np.linspace(0,np.pi, 100)
WArray = W(thetaArray)

k=0
plt.figure(num=k)
plt.plot(thetaArray, WArray)
plt.ylabel(r"$W(\theta)$")
plt.xlabel(r"$\theta$")

plt.savefig("../report/figs/010.pdf", bbox_inches="tight")
plt.close(k)
k += 1
