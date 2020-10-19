import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

M_Z = 91.182    # GeV
Gamma_Z = 2421.75   #   GeV
def A_FB(f, b, sig_f, sig_b):
    a = (f-b)/(f+b)
    sig_a = 2/(f+b)**2 * np.sqrt(b**2*sig_f**2 + f**2*sig_b**2)
    return a, sig_a


data = np.genfromtxt("data", delimiter=",", skip_header=1)
real_data = data[1:, :-1]
#  print(real_data)
cms = real_data[:, 0]
forward = real_data[:, 1]
sigma_forward = np.sqrt(forward)
backward = real_data[:, 2]
sigma_backward = np.sqrt(backward)
#  print(cms, forward, backward)

corr = np.genfromtxt("./cor.dat", delimiter=",", skip_header=1)
print(corr)
if not (np.array_equal(corr[:, 0], cms)):
    print("Error in inputs!")

corr = corr[:, 1]

asym_array, sigma_asym_array = A_FB(forward, backward,
                                    sigma_forward, sigma_backward)
asym_array += corr

sin2thetaW = (1-np.sqrt(asym_array[3]/3))/4
sigma_sin2thetaW = sigma_asym_array[3]/np.sqrt(asym_array[3]) / (8*np.sqrt(3))
print(sin2thetaW, "+-", sigma_sin2thetaW)

k = 0
plt.figure(num=k, figsize=(10, 7))
plt.errorbar(cms, asym_array, color="black", fmt="o",
             yerr=sigma_asym_array, capsize=2)
plt.ylabel("$A_{FB}$")
plt.xlabel(r"$\sqrt{s}$ [GeV]")
plt.savefig("A_FB.pdf", bbox_inches="tight")
plt.close(k)
k += 1


#################################################3
# refined calculation
#################################################3


#  def func_A(sin2, s, A):
#      gA = -1/2
#      gV = -1/2 + 2*sin2
#      ReChi = s*(s-M_Z**2)/((s-M_Z**2)**2 + (s*Gamma_Z/M_Z)**2)
#      f = (3/2*(4*sin2*(1-sin2))*gA**2/(gV**2+gA**2)**2 * ReChi - A)
#      #  print(f, sin2)
#      return f
#
#
#  for i in range(0, len(asym_array)):
#      if i == 3:
#          pass
#      else:
#          root = optimize.fsolve(func_A, 0.234, args=(cms[i], asym_array[i]))
#          print(cms[i], root)
