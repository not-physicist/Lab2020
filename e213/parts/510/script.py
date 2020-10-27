import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2


def GeV_2_nb(x):
    return x/2.5682e-6

def nb_2_GeV(x):
    return x*2.5682e-6

# variable for plot numbering
k = 0

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


def Cov_inv_eps(eps_1, var_eps, a, b, c, d):
    if eps_1.shape == var_eps.shape and eps_1.shape[0] == eps_1.shape[1]:
        N = eps_1.shape[0]
        cov = 0
        for i in range(0, N):
            for j in range(0, N):
                cov += eps_1[a,i] * eps_1[c, i] * var_eps[i, j] * eps_1[j, b] * eps_1[j, d]
        return cov
    else:
        print("Error in input of Cov_inv_eps")
        return ValueError


def Cov_T(M, eps_1, cov_MM, var_eps):
    if eps_1.shape == cov_MM.shape and eps_1.shape[0] == eps_1.shape[0]:
        N = eps_1.shape[0]
        cov = np.zeros((N,N))
        for i in range(0, N):
            for j in range(0, N):
                for alpha in range(0, N):
                    for beta in range(0, N):
                        cov[i,j] += M[alpha] * M[beta] * Cov_inv_eps(eps_1, var_eps, i, alpha, j, beta)

                for k in range(0, N):
                    for l in range(0, N):
                        cov[i,j] += eps_1[i,k] * eps_1[j,l] * cov_MM[k,l]
        return cov
    else:
        print("Error in input of Cov_inv_eps")
        return ValueError


Tdata = data # true counts after eps correction
sigma_Tdata = sigma_data # only the variance/sigma
cov_T = np.zeros((data.shape[0], 4, 4)) # cov. matrix for every cms E

for i in range(0, data.shape[0]):
    # for every cms E
    Tdata[i, :] = np.matmul(inv_eps, data[i, :])
    cov_T[i] = Cov_T(data[i], inv_eps,
                     np.diagflat(sigma_data[i, :]**2), sigma_eps**2)
    #  print(cov_T[i])
    sigma_Tdata[i, :] = np.sqrt(np.diag(cov_T[i]))

print("Corrected counts")
print(Tdata)
print(sigma_Tdata)


#########################################
# calculating partial cross section
#########################################
lumi = np.genfromtxt("./lumi.dat", delimiter=",")[:, 1:]
sigma_lumi = lumi[:, 1]
lumi = lumi[:, 0]
#  print("Lumi", lumi, sigma_lumi)

partial_cross = Tdata
sigma_partial_cross = sigma_Tdata
rad_cor = np.genfromtxt("./rad_corr.dat", delimiter=",", skip_header=1)[:, 1:]
#  print(rad_cor)
cov_partial_cross = cov_T

# correct for efficiency and rad. corrections
for i in range(0, Tdata.shape[0]):
    # for every cms E
    partial_cross[i, :] = Tdata[i, :] / lumi[i]
    sigma_partial_cross[i, :] = np.sqrt((sigma_lumi[i]*Tdata[i, :]/lumi[i]**2)**2
                                        + (sigma_Tdata[i, :]/lumi[i])**2)
    inv_L = np.diagflat([1, 1, 1, 1])/lumi[i]
    var_L = np.diagflat([1, 1, 1, 1])*sigma_lumi[i]**2
    cov_partial_cross[i] = Cov_T(Tdata[i], inv_L, cov_T[i], var_L)
    #  print(cov_partial_cross[i])
    #  print(np.sqrt(np.diag(cov_partial_cross[i])))

    partial_cross[i, 0:3] += rad_cor[i, 1]
    partial_cross[i, 3] += rad_cor[i, 0]


# now cross section
print("Partial cross section")
print(partial_cross)
print(sigma_partial_cross)
np.savetxt("p_cross.dat", partial_cross, delimiter=",", fmt="%.3f")
np.savetxt("sigma_p_cross.dat", sigma_partial_cross, delimiter=",", fmt="%.3f")

# ratio of lep to had
lep_sum_peak = np.sum(partial_cross[3, 0:3])
sigma_lep_sum_peak = np.sqrt(np.sum(sigma_partial_cross[3, 0:3]**2))
print(lep_sum_peak, "+-", sigma_lep_sum_peak)
had_peak = partial_cross[3, 3]
sigma_had_peak = sigma_partial_cross[3, 3]
print(had_peak, "+-", sigma_had_peak)

ratio = had_peak / lep_sum_peak
sigma_ratio = np.sqrt((had_peak/lep_sum_peak**2)**2 * sigma_lep_sum_peak**2
                      + (sigma_had_peak/lep_sum_peak)**2)
print("Exp:", ratio, "+-", sigma_ratio)
print("Theo:", (299*2 + 378*3)/(3*83.8))

#########################################
# breit-wigner fit
#########################################


def Chi2(exp, obs, sig):
    # expectation value and observed
    return np.sum((exp - obs)**2 / sig**2)


def partial_cross_theo(sqrt_s, Gf, Ge, Gz, Mz):
    # return sigma in nb
    # inputs in GeV(2) except Gf, Ge
    s = sqrt_s**2
    Gf /= 1e3 # MeV
    Ge /= 1e3
    sigma_NU = (12*np.pi/Mz**2 * s*Ge*Gf/((s - Mz**2)**2 + (s**2 * Gz**2/Mz**2)))
    return GeV_2_nb(sigma_NU)


partial_cross = partial_cross.T
sigma_partial_cross = sigma_partial_cross.T
#  print(partial_cross)

filenames = ["sigma_ee.pdf", "sigma_mm.pdf", "sigma_tt.pdf", "sigma_qq.pdf"]
legend_array = [r"{ee}", r"{\mu\mu}", r"{\tau\tau}", r"{qq}"]

Gamma_Z_array = np.zeros(4)
sigma_Gamma_Z_array = np.zeros(4)
M_Z_array = np.zeros(4)
sigma_M_Z_array = np.zeros(4)
peak_sigma_array = np.zeros(4)
sigma_peak_sigma_array = np.zeros(4)

for i in [0, 1, 2, 3]:
    popt, pcov = curve_fit(partial_cross_theo, cms, partial_cross[i],
                           p0=(80, 80, 2.4, 90), bounds=([50, 50, 1, 50], [100, 3000, 5, 200]),
                           sigma=sigma_partial_cross[i], absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    print("Parameters", popt, perr)
    Gamma_Z_array[i] = popt[2]
    sigma_Gamma_Z_array[i] = perr[2]
    M_Z_array[i] = popt[3]
    sigma_M_Z_array[i] = perr[3]

    plt.figure(num=k, figsize=(7, 4))
    plt.errorbar(cms, partial_cross[i], yerr=sigma_partial_cross[i], fmt="o",
                 capsize=2, label="data", color="black", ms=2)
    x = np.linspace(np.amin(cms)-0.5, np.amax(cms)+0.5, 300)
    plt.plot(x, partial_cross_theo(x, *popt), label="fit", color="tab:blue")
    upperBound = partial_cross_theo(x, popt[0], popt[1], popt[2]-perr[2], popt[3]-perr[3])
    lowerBound = partial_cross_theo(x, popt[0], popt[1], popt[2]+perr[2], popt[3]+perr[3])
    plt.fill_between(x, lowerBound, upperBound, label="68% CL", color="tab:blue", alpha=0.2)
    plt.xlabel(r"$\sqrt{s} $ [GeV]")
    plt.ylabel(r"$\sigma_" + legend_array[i] + "$ [nb]")
    plt.legend()
    plt.savefig(filenames[i], bbox_inches="tight")
    plt.close(k)
    k += 1

    peak_sigma = np.amax(partial_cross_theo(x, *popt))
    print("Peak sigma:", peak_sigma, "-", peak_sigma - np.amax(lowerBound), "+" , np.amax(upperBound) - peak_sigma)
    peak_sigma_array[i] = peak_sigma
    sigma_peak_sigma_array[i] = ((peak_sigma - np.amax(lowerBound)) + (np.amax(upperBound) - peak_sigma))/2

    # calculate CIs
    #  print(partial_cross_theo(cms, *popt), partial_cross[i])
    chi = Chi2(partial_cross_theo(cms, *popt), partial_cross[i], sigma_partial_cross[i])
    print("Chi2 =", chi)
    df = 3
    print("p =", 1-chi2.cdf(chi, df))

Gamma_Z = np.mean(Gamma_Z_array)
sigma_Gamma_Z = np.sqrt(np.sum(sigma_Gamma_Z_array**2)) / 4
print("Gamma_Z =", Gamma_Z, "+-", sigma_Gamma_Z)
M_Z = np.mean(M_Z_array)
sigma_M_Z = np.sqrt(np.sum(sigma_M_Z_array**2)) / 4
print("M_Z =", M_Z, "+-", sigma_M_Z)

# partial widths
peak_sigma_array = nb_2_GeV(peak_sigma_array) # GeV
sigma_peak_sigma_array = nb_2_GeV(sigma_peak_sigma_array) # GeV

Gamma_f_array = np.zeros(4)
sigma_Gamma_f_array = np.zeros(4)
for i in [0, 1, 2, 3]:
    # partial width
    if i == 0:
        # electron case
        Gamma_f_array[i] = np.sqrt(peak_sigma_array[i]/(12*np.pi)) * M_Z * Gamma_Z * 1e3 # MeV
        sigma_Gamma_f_array[i] =  Gamma_f_array[i] * np.sqrt((sigma_peak_sigma_array[i]/(2*peak_sigma_array[i]))**2
                                           + (sigma_Gamma_Z/Gamma_Z)**2
                                           + (sigma_M_Z/M_Z)**2)
        print("Gamme_e =", Gamma_f_array[i], "+-", sigma_Gamma_f_array[i])
    else:
        Gamma_f_array[i] = peak_sigma_array[i]/(12*np.pi) * M_Z**2 * Gamma_Z**2 / Gamma_f_array[0] * 1e6 # MeV
        sigma_Gamma_f_array[i] = Gamma_f_array[i] * np.sqrt((sigma_peak_sigma_array[i]/peak_sigma_array[i])**2
                                          + (2*sigma_M_Z/M_Z)**2 + (2*sigma_Gamma_Z/Gamma_Z)**2
                                          + (sigma_Gamma_f_array[0]/Gamma_f_array[0])**2)
        print("Gamme_f =", Gamma_f_array[i], "+-", sigma_Gamma_f_array[i])

# neutrino generations
Gamma_inv = Gamma_Z * 1e3 - np.sum(Gamma_f_array) # in MeV
sigma_Gamma_inv = Gamma_inv * np.sqrt((sigma_Gamma_Z/Gamma_Z)**2 + np.sum((sigma_Gamma_f_array/Gamma_f_array)**2))
print("Gamma_inv =", Gamma_inv, "+-", sigma_Gamma_inv)

Gamma_nu = 165.84
nu_gen = Gamma_inv / Gamma_nu
sigma_nu_gen = sigma_Gamma_inv / Gamma_nu
print("Generations of nu =", nu_gen, "+-", sigma_nu_gen)

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
plt.savefig("./sigma_lep.pdf", bbox_inches="tight")
plt.close(k)
k += 1

# plot had sigma
plt.figure(num=k, figsize=(10, 7))
plt.errorbar(cms, partial_cross[-1, :], color="black", capsize=2,
             yerr=sigma_partial_cross[-1, :])
plt.xlabel(r"$\sqrt{s} $ [GeV]")
plt.ylabel(r"$\sigma_{had}$ [nb]")
plt.savefig("./sigma_had.pdf", bbox_inches="tight")
plt.close(k)
k += 1


################################################
# sigma_had, sigma_lep, correlation
################################################
print("Compare sigma_had and sigma_lep")

A = np.array([[1, 1, 1, 0], [0, 0, 0, 1]])
partial_cross_peak = np.matmul(A, partial_cross[:, 3])
var_partial_cross_peak = np.matmul(A, np.matmul(cov_partial_cross[3], A.T))
#  print(cov_partial_cross[3])
#  print(np.sqrt(np.diag(cov_partial_cross[3])))
print(partial_cross_peak, var_partial_cross_peak)

theo_p_cross_peak = np.array([6.280122649121659, 42.02492498544358])
print(theo_p_cross_peak)
