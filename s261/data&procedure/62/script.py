import numpy as np

data = np.genfromtxt("imstats.dat", delimiter="\t")
#  print(data)

sigma_RON = data[0, 6]
print("RON sigma =", sigma_RON)

sigma_e = np.sqrt((data[1, 6]**2 - 2*sigma_RON**2)/2)
print("Photon sigma =", sigma_e)

sigma_PRNU = np.array([data[2, 6], data[3, 6]])
sigma_PRNU = np.sqrt(sigma_PRNU**2 - sigma_e**2 - sigma_RON**2)
sigma_PRNU = 1/2*np.sqrt(sigma_PRNU[0]**2 + sigma_PRNU[1]**2)
print("PRNU sigma =", sigma_PRNU)

signal_level = np.array([data[2, 5], data[3, 5]])
sigma_signal_level = np.std(signal_level)
signal_level = np.mean(signal_level)
print("Signal level =", signal_level, "+-", sigma_signal_level)

f_PRNU = sigma_PRNU / signal_level
sigma_f_PRNU = f_PRNU * sigma_signal_level/signal_level
print("PRNU factor =", f_PRNU, "+-", sigma_f_PRNU)

gain = signal_level / sigma_e**2
sigma_gain = gain * sigma_signal_level/signal_level
print("gain =", gain, "+-", sigma_gain)
print("RON sigma in e- =", gain*sigma_RON, "+-", sigma_gain*sigma_RON)
