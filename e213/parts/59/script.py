import numpy as np

data = np.genfromtxt("N.dat", skip_header=2, delimiter=",").T[1:].T
#  print(data)

N_orig = data[1]
N_ee = data[2]
N_mm = data[3]
N_tt = data[4]
N_qq = data[5]

N = np.array([N_orig, N_ee, N_mm, N_tt, N_qq])
sigma_N = np.sqrt(N)

eps = np.array([N_ee, N_mm, N_tt, N_qq])/N_orig
sigma_eps = np.sqrt((sigma_N[0]*N[1:]/N[0]**2)**2 + (sigma_N[1:]/N[0])**2)

#########################################
# correct ee efficiency
#########################################


def ind_int(x):
    return (x+x**3/3)


def s_int_result(a, b):
    return (ind_int(b) - ind_int(a))


corr_factor = s_int_result(-1, 1) / s_int_result(-0.9, 0.5)
print("Correction factor for N_ee:", corr_factor)

eps[:, 0] = eps[:, 0]/corr_factor
sigma_eps[:, 0] = sigma_eps[:, 0]/corr_factor


np.set_printoptions(precision=5, suppress=True)
np.savetxt("eps.dat", eps, delimiter=',', fmt="%.4f")
np.savetxt("sigma_eps.dat", sigma_eps, delimiter=',', fmt="%.4f")
