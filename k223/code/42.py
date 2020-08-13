# python script for section 4.2
import numpy as np
import matplotlib.pyplot as plt


print("Predicted coefficients")
A22 = 0.1020
A44 = 0.0091
Ap = 1 - A22/2 + 3/8*A44
Bp = 3/2 * A22 - 15/4 * A44
Cp = 35/8 *A44
print("A' = %f, B'=%f, C'=%f" % (Ap, Bp, Cp))


B = Bp/Ap
C = Cp/Ap
print("scaled: B=%f, C=%f" % (B, C))
print("other definition: B=%f, C=%f" % (1/8, 1/24))

alpha = B + C
beta = B- C
print("alpa=%f, beta=%f" % (alpha, beta))

def f(theta, A, alpha, beta):
    B = (alpha + beta)/2
    C = (alpha - beta)/2
    return A*(1 + B * np.cos(theta)**2 + C * np.cos(theta)**4 )

thetaArray = np.linspace(0, np.pi, 100)

addArray = np.array([[0,0],[0.1,0], [0, 0.2]])
for i in [0,1,2]:
    a = alpha + addArray[i, 0]
    b = beta + addArray[i, 1]
    fArray = f(thetaArray, 1 , a, b)
    plt.plot(thetaArray, fArray, label=r"$\alpha="
             + "{:.3f}".format(a) + r", \beta="
             + "{:.3f}".format(b)
             + "$")

plt.legend()
plt.xlabel(r"$\theta$")
plt.ylabel(r"$f(\theta)/A$")
plt.savefig("../report/figs/alpha-beta.pdf", bbox_inches="tight")
plt.close()
