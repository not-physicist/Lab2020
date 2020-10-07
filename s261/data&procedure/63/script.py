import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def Sort2Array(x, y):
    temp = np.array([x, y]).T
    temp = temp[temp[:, 0].argsort()].T
    return temp[0], temp[1]


def Sort3Array(x, y, z):
    temp = np.array([x, y, z]).T
    temp = temp[temp[:, 0].argsort()].T
    return temp[0], temp[1], temp[2]


def Line(x, a, b):
    return a*x + b


def Inverte_line(y, a, b):
    return (y - b) / a


filename = "./imstats.dat"
header = 6  # number of header rows
exptime = np.array([])
signal_level = np.array([])
sigma_signal_level = np.array([])

diff_exptime = np.array([])
diff_variance = np.array([])

with open(filename, newline="") as csvfile:
    a = csv.reader(csvfile, delimiter='\t')
    # a is a object containing row arrays
    for row in a:
        #  print(header)
        if header <= 0:
            if "diff" not in row[0]:
                # only orig images
                row[0] = row[0].replace("FLATLIN_", "")
                row[0] = row[0].replace("-1", "")
                row[0] = row[0].replace(".fits", "")
                row[0] = row[0].replace("s", "")
                exptime = np.append(exptime, float(row[0]))

                # get rid of space
                for i in range(1, len(row)):
                    # except the file name
                    row[i] = row[i].replace(" ", "")

                # read mean value
                signal_level = np.append(signal_level, float(row[5]))
                sigma_signal_level = np.append(sigma_signal_level,
                                               float(row[6]))
                #  print(row)
            else:
                #  print(row)
                # only diff images
                row[0] = row[0].replace("FLATLIN_", "")
                row[0] = row[0].replace("-1", "")
                row[0] = row[0].replace("_diff.fits", "")
                row[0] = row[0].replace("s", "")
                diff_exptime = np.append(diff_exptime, float(row[0]))

                for i in range(1, len(row)):
                    # except the file name
                    row[i] = row[i].replace(" ", "")
                diff_variance = np.append(diff_variance, float(row[6]))

        header -= 1

exptime, signal_level, sigma_signal_level = Sort3Array(exptime,
                                                       signal_level,
                                                       sigma_signal_level)

#  print(exptime)
#  print(signal_level)
#  print(sigma_signal_level)

k = 0
color="tab:blue"
plt.figure(num=k, figsize=(10, 7))
plt.errorbar(exptime, signal_level, yerr=sigma_signal_level,
             fmt="o", label="data", color=color)

# fit
mask = (exptime < 1.05)
fit_exptime = exptime[mask]
fit_signal_level = signal_level[mask]
fit_sigma_signal_level = sigma_signal_level[mask]
#  print(fit_exptime, fit_signal_level)
popt, perr = curve_fit(Line, fit_exptime,
                       fit_signal_level, sigma=fit_sigma_signal_level)
print("Best fit parameter and covariance matrix:", popt, perr)

x = np.linspace(np.amin(fit_exptime) - 0.1, np.amax(fit_exptime) + 0.1, 100)
plt.plot(x, Line(x, *popt), label="fit", color=color)

ax = plt.gca()
plt.xlabel("exposure time/s")
ax.set_ylabel("mean pixel values / ADU", color=color)

# residual
color = "tab:red"
pred_signal = Line(exptime, *popt)
res = pred_signal - signal_level
ax2 = ax.twinx()
ax2.scatter(exptime, res, color=color)
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_ylabel("residual / ADU", color=color)

ax.legend(loc=2)
plt.savefig("exp_mean.pdf", bbox_inches="tight")
plt.close(k)
k += 1

pixel_max = np.amax(signal_level)
#  print(pixel_max)
exptime_max = Inverte_line(pixel_max, *popt)
jac = - 1/popt[0] * np.array([exptime_max, 1])
sigma_exptime_max = jac.dot(perr.dot(jac.transpose()))
print("Maximal exposure time=", exptime_max, "+-", sigma_exptime_max)


##################################################################
# difference
##################################################################

diff_exptime, diff_variance = Sort2Array(diff_exptime, diff_variance)
diff_variance = diff_variance**2

signal_for_diff = np.array([])
sigma_signal_for_diff = np.array([])
length = (len(signal_level))/2
for i in range(0, int(length)):
    mean = np.mean([signal_level[2*i], signal_level[2*i+1]])
    signal_for_diff = np.append(signal_for_diff, mean)

    std = np.std([signal_level[2*i], signal_level[2*i+1]])
    sigma_signal_for_diff = np.append(sigma_signal_for_diff, std)

#  print(signal_for_diff)
sigma_signal_level = sigma_signal_level[::2]

# fit
mask = (diff_variance > 0)  # mask for only linear region
popt, perr = curve_fit(Line, signal_for_diff[mask], diff_variance[mask])
print("Fit parameters and covariance matrix", popt, perr)

plt.figure(num=k, figsize=(10, 7))

x = np.linspace(np.amin(signal_for_diff)-1000, np.amax(signal_for_diff), 100)
plt.errorbar(signal_for_diff, diff_variance, color="black", label="data",
             xerr=sigma_signal_level, fmt="o")
plt.plot(x, Line(x, *popt), color="black", label="linear fit")

mask_signal, mask_variance = signal_for_diff[mask], diff_variance[mask]
#  anno_point = (mask_signal[-1], mask_variance[-1]-3)
#  anno_text = (mask_signal[-1], mask_variance[-1]-50)
#  plt.annotate("saturation", xy=anno_point, xytext=anno_text,
             #  arrowprops=dict(facecolor='black', shrink=0.05))
inv_mask_signal = signal_for_diff[~mask]

sat = np.mean([inv_mask_signal[0], mask_signal[-1]])
sigma_sat = np.fabs(inv_mask_signal[0] - mask_signal[-1])
#  print(signal_for_diff)
print("Saturation at:", sat, "+-", sigma_sat, "ADU")

plt.xscale("log")
plt.xlabel("signal of original images / ADU")
plt.yscale("log")
plt.ylabel(r"variance of difference frames / ADU$^2$")
plt.legend()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig("diff.pdf", bbox_inches="tight")
plt.close(k)
k += 1

gain = 2/popt[0]
sigma_gain = np.fabs( 2 / popt[0]**2 * np.sqrt(np.diag(perr))[0])
print("Gain = ", gain, "+-", sigma_gain, "e/ADU")

full_well = gain * sat
sigma_full_well = full_well * np.sqrt((sigma_gain/gain)**2 + (sigma_sat/sat)**2)
print("Full well capacity =", full_well, "+-", sigma_full_well, "e-")
