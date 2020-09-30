import numpy as np
from lensing_analysis import Pixel_2_distance_arcsec

MFWHM = np.genfromtxt("list.dat").T[-1]
print(MFWHM)
MFWHM = MFWHM[:-1]
print(MFWHM)
seeing, sigma_seeing = np.mean(MFWHM), np.std(MFWHM)
print(seeing, sigma_seeing)
seeing, sigma_seeing = Pixel_2_distance_arcsec(seeing), Pixel_2_distance_arcsec(sigma_seeing)
print(seeing, sigma_seeing)
