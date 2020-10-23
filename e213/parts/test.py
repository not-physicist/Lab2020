import numpy as np

sin2_thetaW = 0.23
thetaW = np.arcsin(np.sqrt(sin2_thetaW))
a = -1/2/(2*np.sin(thetaW)*np.cos(thetaW))
v = (-1/2 + 2 * sin2_thetaW)/ (2*np.sin(thetaW)*np.cos(thetaW))
pre = 3/2 * a**2 / ((v**2+a**2)**2)
print(pre)

Gamma_Z = 2.537
M_Z = 91.1876
s = 91.225**2
post = s*(s-M_Z**2) / ((s-M_Z**2)**2 + (s*Gamma_Z/M_Z)**2)
print(post)
print(pre*post)
