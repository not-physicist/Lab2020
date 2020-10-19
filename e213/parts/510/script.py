import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# variable for plot numbering
k = 0

def Cov_inv_matrix(mat, sig_mat, inv_mat):
    A = np.matmul(inv_mat, inv_mat)
    B = np.matmul(sig_mat, sig_mat)
    return np.matmul(A, np.matmul(B, A))


data = np.genfromtxt("data_new", delimiter=",")
# cuts: , none, ee, mm, tt, qq
cms = data[1:, 0]
print(cms)
data = data[1:, 2:]
sigma_data = np.sqrt(data)
print("Raw counts")
print(data, sigma_data)

eps = np.genfromtxt("../59/eps.dat", delimiter=",")
sigma_eps = np.genfromtxt("../59/sigma_eps.dat", delimiter=",")
inv_eps = inv(eps)

#  print(eps, sigma_eps)
#  print("Inverse detection matrix")
#  print(inv_eps)
#  print("+-", sigma_inv_eps)
#  print(np.matmul(eps, inv(eps)))


#########################################
# correct for efficiency
#########################################
# true counts
Tdata = data
sigma_Tdata = sigma_data
for i in range(0, data.shape[0]):
    # for every cms E
    Tdata[i, :] = np.matmul(inv_eps, data[i, :])
    sigma_Tdata[i, :] = np.sqrt(np.diag(np.matmul(inv_eps, np.matmul(np.diagflat(data[i, :]), inv_eps.T))))

print("Corrected counts")
print(Tdata)
print(sigma_Tdata)


#########################################
# calculating partial cross section
#########################################
lumi = np.genfromtxt("./lumi.dat", delimiter=",")[:, 1:]
sigma_lumi = lumi[:, 1]
lumi = lumi[:, 0]
#  print(lumi, sigma_lumi)

partial_cross = Tdata
sigma_partial_cross = sigma_Tdata
rad_cor = np.genfromtxt("./rad_corr.dat", delimiter=",", skip_header=1)[:, 1:]
#  print(rad_cor)

# correct for efficiency and rad. corrections
for i in range(0, Tdata.shape[0]):
    # for every cms E
    partial_cross[i, :] = Tdata[i, :] / lumi[i]
    sigma_partial_cross[i, :] = np.sqrt((sigma_lumi[i]*Tdata[i, :]/lumi[i]**2)**2 + (sigma_Tdata[i, :]/lumi[i])**2)
    partial_cross[i, 0:3] += rad_cor[i, 1]
    partial_cross[i, 3] += rad_cor[i, 0]


# now cross section
print("Partial cross section")
print(partial_cross)
print(sigma_partial_cross)
np.savetxt("p_cross.dat", partial_cross, delimiter=",")
np.savetxt("sigma_p_cross.dat", partial_cross, delimiter=",")


#########################################
# breit-wigner fit
#########################################

def partial_cross_theo(sqrt_s, Gf, Ge, Gz, Mz):
    # return sigma in nb
    # inputs in GeV(2)
    s = sqrt_s**2
    Gf /= 1e3
    Ge /= 1e3
    sigma_NU = (12*np.pi/Mz**2 * s*Ge*Gf/((s - Mz**2)**2 + (s**2 * Gz**2/Mz**2)))
    return sigma_NU/2.5682e-6

partial_cross = partial_cross.T
sigma_partial_cross = sigma_partial_cross.T

filenames = ["sigma_ee.pdf", "sigma_mm.pdf", "sigma_tt.pdf", "sigma_qq.pdf"]
for i in [0, 1, 2, 3]:
    popt, pcov = curve_fit(partial_cross_theo, cms, partial_cross[i],
                           p0=(80, 80, 2.4, 90),
                           sigma=sigma_partial_cross[i], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    print(popt)
    print(perr)
    plt.figure(num=k, figsize=(7, 4))
    plt.errorbar(cms, partial_cross[i], yerr=sigma_partial_cross[i], fmt="o",
                 capsize=2, label="data", color="black")
    x = np.linspace(np.amin(cms)-0.5, np.amax(cms)+0.5, 300)
    plt.plot(x, partial_cross_theo(x, *popt), label="fit", color="tab:blue")
    lowerBound = partial_cross_theo(x, popt[0], popt[1], popt[2]-perr[2], popt[3]-perr[3])
    upperBound = partial_cross_theo(x, popt[0], popt[1], popt[2]+perr[2], popt[3]+perr[3])
    plt.fill_between(x, lowerBound, upperBound, label="68% CL", color="tab:blue", alpha=0.2)
    plt.xlabel(r"$\sqrt{s} $ [GeV]")
    plt.ylabel(r"$\sigma_i$ [nb]")
    plt.legend()
    plt.savefig(filenames[i], bbox_inches="tight")
    plt.close(k)
    k += 1

# plot lept. sigmas
plt.figure(num=k, figsize=(10, 7))
legend_array = [r"$ee$", r"$\mu\mu$", r"$\tau\tau$"]
for i in range(0, partial_cross.shape[0]-1):
    # for every cms E
    # only leptonic
    plt.errorbar(cms, partial_cross[i, :],
                 label=legend_array[i],
                 yerr=sigma_partial_cross[i, :], capsize=2)

plt.xlabel(r"$\sqrt{s} $ [GeV]")
plt.ylabel(r"$\sigma_i$ [nb]")
plt.legend()
plt.savefig("./diff_cross_lep.pdf", bbox_inches="tight")
plt.close(k)
k += 1

# plot had sigma
plt.figure(num=k, figsize=(10, 7))
plt.errorbar(cms, partial_cross[-1, :], color="black",
             yerr=sigma_partial_cross[-1, :])
plt.xlabel(r"$\sqrt{s} $ [GeV]")
plt.ylabel(r"$\sigma_i$ [nb]")
plt.savefig("./diff_cross_had.pdf", bbox_inches="tight")
plt.close(k)
k += 1
