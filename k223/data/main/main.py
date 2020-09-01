import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse

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

popt, pcov = curve_fit(fit, angle, coincRate, sigma=sigmaCoincRate,
                       p0=[45, 45/8, 45/24])
print(pcov)
sigmaPara = np.sqrt(np.diagonal(pcov))
print("Fit parameters: A=%f +- %f, B=%f +- %f, C=%f +- %f"
      % (popt[0], sigmaPara[0], popt[1], sigmaPara[1], popt[2], sigmaPara[2]))

# plot fit curve
angleCont = np.linspace(np.amin(angle), np.amax(angle), 100)
plt.plot(angleCont, fit(angleCont, *popt), label="fit", color="tab:blue")

# draw confidence interval
#  lowerBound = fit(angleCont, *(popt - sigmaPara))
#  upperBound = fit(angleCont, *(popt + sigmaPara))
#  plt.fill_between(angleCont, lowerBound, upperBound,
                 #  label="68% CL", color="tab:blue", alpha=0.2)

# fit using only ^2 term
def fitB(angle, A, B):
    theta = angle *np.pi/180
    return A* (1 + B*np.cos(theta)**2)

poptB, pcovB = curve_fit(fitB, angle, coincRate, sigma=sigmaCoincRate,
                       p0=[45, 45/8])
print("Using alternative fit function")
print(poptB)
print(np.sqrt(np.diag(pcovB)))
#  angleCont = np.linspace(np.amin(angle), np.amax(angle), 100)
#  plt.plot(angleCont, fitB(angleCont, *poptB), label="fitB", color="tab:blue")

# apply correction to the prediction curve
Q2 = np.mean([0.9338, 0.9343])
Q4 = np.mean([0.7913, 0.7934])
print("Q2=%f, Q4=%f" % (Q2, Q4))
QMatrix = np.array([[1/Q2, 0], [0, 1/Q4]]) # matrix contains diagonal 1/Q_i

A22 = 0.1020 * Q2
A44 = 0.0091 * Q4
Ap = 1 - A22/2 + 3/8*A44
Bp = 3/2 * A22 - 15/4 * A44
Cp = 35/8 *A44
B = Bp/Ap
C = Cp/Ap
print("Prediction: B = %f, C = %f" % (B,C))

plt.plot(angleCont, fit(angleCont, popt[0], B, C),
         label="prediction", color="tab:orange")
# draw confidence interval
#  lowerBound = fit(angleCont, popt[0] + sigmaPara[0], B, C)
#  upperBound = fit(angleCont, popt[0] - sigmaPara[0], B, C)
#  plt.fill_between(angleCont, lowerBound, upperBound,
                 #  label="68% CL", color="tab:orange", alpha=0.2)

plt.legend()
plt.ylabel("coincidence rate[$s^{-1}$]")
plt.xlabel(r"$\theta$[°]")
plt.savefig("../../report/figs/angCor.pdf", bbox_inches="tight")

plt.close(k)
k += 1

# convert to alpha, beta
# measurement
BCArray = np.array([popt[1], popt[2]])
matrixM = np.array([[1,1],[1,-1]])
alphaBetaArray = np.matmul(matrixM, BCArray)
print("Alpha beta measure:", alphaBetaArray)
sigmaAlphaBetaArray = np.matmul(matrixM, np.matmul(pcov[1:, 1:], matrixM.T))
print("covariance matrix:", sigmaAlphaBetaArray)

BCArrayTheo = np.array([B, C])
alphaBetaArrayTheo = np.matmul(matrixM, BCArrayTheo)
print("Alpha beta theo:", alphaBetaArrayTheo)


# calculate measured A22, A44
Bexp = popt[1]
Cexp = popt[2]
matrixA = np.array([[(B+3)/2,  (-30-3*B)/8], [C/2, (-3*C + 35)/8]])
matrixB = np.array([Bexp, Cexp])
matrixX = np.matmul(np.linalg.inv(matrixA), matrixB)
matrixX = np.matmul(matrixX, QMatrix)
print("Measured coefficients without errors:\nA22 = %f, A44 = %f"
      % (matrixX[0], matrixX[1]))

# jacobian computed by mathematica
temp = (8-4*matrixX[0]+3*matrixX[1])**2
jac = np.array([[96-84*matrixX[1], (-240 + 84*matrixX[0])],
                [140*matrixX[1], -140*(-2 + matrixX[0])]])/temp
covAkk = np.matmul(np.matmul(np.linalg.inv(jac), pcov[1:, 1:]), np.linalg.inv(jac).T )
print(np.sqrt(np.diag(covAkk)))

# plot for parameters in 2d plane (B, C)

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_patch(ellip)
    #  ellip.set(label=str(nstd)+"$\sigma$")
    return ellip

def plot_ellipses(points, max_nstd=2, **kwargs):
    for i in range(0, max_nstd+1):
        plot_point_cov(points, nstd=i, **kwargs)

xlims = np.array([0.06, 0.18])
ylims = np.array([-0.03, 0.05])

plt.figure(num=k, figsize=(10,5))

muArray = np.array([Bexp,Cexp])
print("Bexp = %f, Cexp = %f" % (muArray[0], muArray[1]))
covBC = pcov[1:, 1:]
#  print(covBC)
points = np.random.multivariate_normal(
        mean=(muArray[0], muArray[1]), cov=covBC, size=1000)

plot_ellipses(points, 5, alpha=0.3, color="tab:blue")

plt.scatter([B], [C], color="black", alpha=1, label="theory")
plt.scatter([muArray[0]], [muArray[1]], color="black", alpha=1, marker="x", label="measure")

plt.xlim(*xlims)
plt.ylim(*ylims)
plt.xlabel("B")
plt.ylabel("C")
plt.legend()

plt.savefig("../../report/figs/BCpara.pdf", bbox_inches="tight")
plt.close(k)
k += 1

# plot to compare angular asymmetry
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

# prediction of random coincidence rate
resT = 25e-9
C1 = np.mean(LcountRate)
sigmaC1 = np.std(LcountRate)
C2 = np.mean(RcountRate)
sigmaC2 = np.std(RcountRate)
print("Count rates: %f +- %f, %f +- %f" % (C1, sigmaC1, C2, sigmaC2))
Rdot = C1 * C2 * resT
sigmaRdot = np.sqrt((C2 * sigmaC1) ** 2 + (C1 * sigmaC2)**2) * resT
print("Predicted random coincidence rate: %f +- %f" % (Rdot, sigmaRdot))
