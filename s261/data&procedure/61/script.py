import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

Eg = 1.16   # eV
kB = 8.62e-5    # eV/K
gain = np.mean([1.514, 1.47])
sigma_gain = 0.05
print("gain =", gain, "+-", sigma_gain)

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


def Dark_func(T, c, A):
    # T in Celsius
    temp = T + 273.15
    return c * np.power(temp, 3/2) * np.exp(-Eg / (2 * kB * temp)) + A


filename = "./imstats.new.dat"
header = 6  # number of header rows
temperature = -100
temp_array = np.array([])
dark_current = np.array([])
sigma_dark_current = np.array([])
bias = np.array([])
sigma_bias = np.array([])

with open(filename, newline="") as csvfile:
    a = csv.reader(csvfile, delimiter='\t')
    # a is a object containing row arrays
    for row in a:
        #  print(header)
        if header <= 0:
            row[0] = row[0].replace("DARK_", "")
            row[0] = row[0].replace(".fits", "")
            numbers = row[0].split("deg_")
            numbers[1] = numbers[1].replace("s", "")
            #  print(numbers)
            if numbers[0] != temperature:
                # empty the arrays
                exptime = np.array([])
                signal_level = np.array([])
                sigma_signal_level = np.array([])
                temperature = numbers[0]

            exptime = np.append(exptime, float(numbers[1]))

            # get rid of space
            for i in range(1, len(row)):
                # except the file name
                row[i] = row[i].replace(" ", "")

            # read mean value
            signal_level = np.append(signal_level, float(row[5]))
            sigma_signal_level = np.append(sigma_signal_level,
                                           float(row[6]))

            if len(exptime) == 3:
                # ready to process
                #  print("Temperature =", temperature, "C")
                #  print(exptime)
                #  print(signal_level)
                #  print(sigma_signal_level)
                popt, perr = curve_fit(Line, exptime,
                                       signal_level, sigma=sigma_signal_level)
                sigma = np.sqrt(np.diag(perr))
                #  print(popt, perr)
                dark_current = np.append(dark_current, popt[0])
                sigma_dark_current = np.append(sigma_dark_current, sigma[0])
                bias = np.append(bias, popt[1])
                sigma_bias = np.append(sigma_bias, sigma[1])
                temp_array = np.append(temp_array, float(temperature))
        header -= 1

np.savetxt("fit.dat", np.array([temp_array, dark_current,
                                sigma_dark_current, bias, sigma_bias]).T,
           delimiter=",")

temp_array_sorted, dark_current, sigma_dark_current = Sort3Array(temp_array,
                                                            dark_current,
                                                            sigma_dark_current)
temp_array_sorted, bias, sigma_bias = Sort3Array(temp_array, bias, sigma_bias)

dark_current *= gain
sigma_dark_current = np.sqrt((gain*sigma_dark_current)**2
                             + (dark_current*sigma_gain)**2)

k = 0
plt.figure(num=k, figsize=(10, 7))
plt.errorbar(temp_array_sorted, dark_current, color="black",
             yerr=sigma_dark_current, fmt="o", label="data")

# fit
popt, perr = curve_fit(Dark_func, temp_array_sorted, dark_current,
                       sigma=sigma_dark_current, p0=(1e-1, 1e-2))
perr = np.sqrt(np.diag(perr))
print("c =", popt[0], "+-", perr[0])
c = popt[0]
sigma_c = perr[0]
print("A =", popt[1], "+-", perr[1])
x = np.linspace(np.amin(temp_array_sorted)-2, np.amax(temp_array_sorted), 200)
plt.plot(x, Dark_func(x, *popt), color="black", label="fit")

plt.xlabel("set temperature/C")
plt.ylabel("dark current/ $e^-$pix$^{-1} s^{-1}$")
plt.legend(loc=2)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("darkCurrent.pdf", bbox_inches="tight")
plt.close(k)
k += 1

plt.figure(num=k, figsize=(10, 7))
plt.errorbar(temp_array_sorted, bias, yerr=sigma_bias, fmt="o", color="black")
plt.xlabel("set temperature/C")
plt.ylabel("bias/ADU")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("bias.pdf", bbox_inches="tight")
plt.close(k)
k += 1

I25 = gain * Dark_func(-25, *popt)
sigma_I25 = np.fabs(I25 * np.sqrt((c/sigma_c)**2 + (gain/sigma_gain)**2))
print("Dark current at -25C = ", I25, "+-", sigma_I25)

I_noise = -300 * I25/gain
print(I_noise)
