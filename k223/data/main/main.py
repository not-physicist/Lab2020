import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# random coincidence
data = np.genfromtxt("./random.dat", skip_header=2, delimiter=",").T
randomCoincRate = data[2]/data[0]
sigmarandomCoincRate = np.sqrt(data[2])/data[0]
print("random coincidence rate is:", randomCoincRate, "+-",sigmarandomCoincRate)

# read data, save to arrays
data = np.genfromtxt("main.dat", skip_header=3, delimiter=",").T
angle = data[0]
duration = data[1]
Lcounter = data[2]
coinCounter = data[3]
Rcounter = data[4]
sigmaTheta = 2 # degree

# calculate L and R count rate
LcountRate = Lcounter/duration
sigmaLcountRate = np.sqrt(Lcounter)/duration
RcountRate = Rcounter/duration
sigmaRcountRate = np.sqrt(Rcounter)/duration

# plot for raw count rates
k = 0
plt.figure(num=k, figsize=(10,5))

plt.errorbar(angle, LcountRate, yerr=sigmaLcountRate, label="fixed",
             xerr=sigmaTheta, fmt="o", color="grey")
plt.errorbar(angle, RcountRate, yerr=sigmaRcountRate, label="mobile",
             xerr=sigmaTheta, fmt="o", color="black")

plt.ylabel("count rate[$s^{-1}$]")
plt.xlabel(r"$\theta$[°]")
plt.legend()
plt.savefig("../../report/figs/countRate.pdf", bbox_inches="tight")

plt.close(k)
k += 1

# correction for misalignment
normArray = LcountRate[0] / LcountRate # normalization factor
# ignore the error in normalization factor, really really tiny
#  print(normArray)

# calculate coincidence rate
coincRate = coinCounter/duration
# first subtract random coincidence, then normalize to counter misalignment
coincRate -= randomCoincRate
coincRate *= normArray

# systematics
mask = (angle == 180)
coincRate180 = coincRate[mask]
#  print("coincidence rate at pi after subtracting random coincidence and correcting for misalignment",
      #  coincRate180)
sigmaSys = np.std(coincRate180)
print("systematical error in count rate:", sigmaSys)

# propagated error + systematics
sigmaCoincRate = np.sqrt((np.sqrt(coinCounter)/duration)**2
                         + sigmarandomCoincRate**2) + sigmaSys

# figure of angular correlation
k = 0
plt.figure(num=k, figsize=(10,5))
plt.errorbar(angle, coincRate, label="data", color="black", capsize=2,
             yerr=sigmaCoincRate, xerr=sigmaTheta, fmt="o")

# fit
def fit(angle, A, B, C):
    theta = angle *np.pi/180
    return A* (1 + B*np.cos(theta)**2 + C*np.cos(theta)**4 )

popt, pcov = curve_fit(fit, angle, coincRate, sigma=sigmaCoincRate, p0=[45, 45/8, 45/24])
sigmaPara = np.sqrt(np.diagonal(pcov))
print("Fit parameters: A=%f +- %f, B=%f +- %f, C=%f +- %f"
      % (popt[0], sigmaPara[0], popt[1], sigmaPara[1], popt[2], sigmaPara[2]))

angleCont = np.linspace(np.amin(angle), np.amax(angle), 100)
plt.plot(angleCont, fit(angleCont, *popt), label="fit", color="tab:blue")

# draw confidence interval
lowerBound = fit(angleCont, *(popt - sigmaPara))
upperBound = fit(angleCont, *(popt + sigmaPara))
plt.fill_between(angleCont, lowerBound, upperBound,
                 label="68% CL", color="tab:blue", alpha=0.2)

# apply correction to the prediction curve
Q2 = np.mean([0.9338, 0.9343])
Q4 = np.mean([0.7913, 0.7934])
print("Q2=%f, Q4=%f" % (Q2, Q4))
A22 = 0.1020 * Q2
A44 = 0.0091 * Q4
Ap = 1 - A22/2 + 3/8*A44
Bp = 3/2 * A22 - 15/4 * A44
Cp = 35/8 *A44
B = Bp/Ap
C = Cp/Ap

plt.plot(angleCont, fit(angleCont, popt[0], B, C),
         label="prediction", color="tab:orange")
# draw confidence interval
lowerBound = fit(angleCont, popt[0] + sigmaPara[0], B, C)
upperBound = fit(angleCont, popt[0] - sigmaPara[0], B, C)
plt.fill_between(angleCont, lowerBound, upperBound,
                 label="68% CL", color="tab:orange", alpha=0.2)

plt.legend()
plt.ylabel("coincidence rate[$s^{-1}$]")
plt.xlabel(r"$\theta$[°]")
plt.savefig("../../report/figs/angCor.pdf", bbox_inches="tight")

plt.close(k)
k += 1

# compare angular asymmetry
maskG180 = (angle > 180)
angleG180 = angle[maskG180]
coincRateG180 = coincRate[maskG180]
sigmaCoincRateG180 = sigmaCoincRate[maskG180]

maskL180 = (angle <= 180)
angleL180 = angle[maskL180]
coincRateL180 = coincRate[maskL180]
sigmaCoincRateL180 = sigmaCoincRate[maskL180]

plt.figure(num=k, figsize=(10,5))

ax1 = plt.gca()
ax2 = ax1.twiny()
ax2.spines['top'].set_color('tab:blue')
ax2.tick_params(axis='x', colors='tab:blue')
ax2.set_xlabel(r"$\theta$[°] if $\theta > 180°$", color="tab:blue")
ax2.set_xlim(305, 175)

ax1.set_xlabel(r"$\theta$[°] if $\theta < 180°$")
ax1.set_xlim(55, 185)

ax1.set_ylabel("coincidence rate[$s^{-1}$]")

ax1.errorbar(angleL180, coincRateL180, fmt="o", capsize=3, color="black",
             yerr=sigmaCoincRateL180, xerr=sigmaTheta)
ax2.errorbar(angleG180, coincRateG180, fmt="o", capsize=3, color="tab:blue",
             yerr=sigmaCoincRateG180, xerr=sigmaTheta)

plt.savefig("../../report/figs/angAsymm.pdf", bbox_inches="tight")
plt.close(k)
k += 1
