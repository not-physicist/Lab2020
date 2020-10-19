import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

p_cross = np.genfromtxt("../510/p_cross.dat", delimiter=",").T
#  print(p_cross)
sigma_p_cross = np.genfromtxt("../510/sigma_p_cross.dat", delimiter=",").T



