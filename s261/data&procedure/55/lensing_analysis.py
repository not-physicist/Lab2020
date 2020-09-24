import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# Hubble time and constant
tH = 4.55e17 #s
H0 = 1/tH #s^{-1}
c = 3e8
G = 6.67e-11 #m3 kg-1 s-2

# redshift and so on
zs = 1.547
zd = 0.577
omega_m = 0.3089
omega_Lambda = 0.6911
omega_rad = 1 - omega_m - omega_Lambda
w_zd = 0.4985 * c / H0
w_zs = 1.039 * c / H0

def f_K(omega):
    # curvature 0
    return omega

def Integrand_angular_D(x):
    # integrand in Angular_D
    return (omega_rad*(1+x)**2 + omega_m*(1+x)**3 + omega_Lambda)**(-1/2)

def Angular_D(z1, z2):
    # gives angular distance
    temp = c * tH * integrate.quad(Integrand_angular_D, z1, z2)[0]
    return 1/(1+z2) * f_K(temp)

def Inverted_Einstein_radius(thetaE, sigma_thetaE):
    # gives velocity dispersion in unit of c
    disp = np.sqrt(thetaE * D_s / (D_ds * 4*np.pi))
    sigma_disp = np.fabs(disp/2/thetaE*sigma_thetaE)
    return disp, sigma_disp

def Pixel_2_distance(pix):
    # convert pixels to physical distance
    # 0.4'' per pixel
    phy_dist = 0.4 * pix # arcsec
    phy_dist /= 3600 # degree
    phy_dist *= np.pi/180 # radians

    return phy_dist

coor_image1 = np.array([29.69, 24.71])
sigma_coor_image1 = np.array([0.05, 0.08])
coor_image2 = np.array([31.11, 31.16])
sigma_coor_image2 = np.array([0.01, 0.01])

def Pixel_distance(a, sigma_a, b, sigma_b):
    delta_x =  np.sqrt((a[0]-b[0])**2 + (a[1] - b[1])**2)
    delta_x_a = (a[0] - b[0])/delta_x
    delta_y_a = (a[1] - b[1])/delta_x
    delta_x_b = (-a[0] + b[0])/delta_x
    delta_y_b = (-a[1] + b[1])/delta_x
    sigma_delta_x = np.sqrt( delta_x_a**2 * sigma_a[0]**2 + delta_y_a**2 * sigma_a[1]**2
                            + delta_x_b**2 * sigma_b[0]**2 + delta_y_b**2 * sigma_b[1]**2)
    return delta_x, sigma_delta_x

if __name__ == "__main__":
    D_ds = Angular_D(zd, zs)
    D_s = Angular_D(0, zs)
    D_d = Angular_D(0, zd)
    print("Angular distances:", D_ds, D_s, D_d)

    separation, sigma_separation = Pixel_distance(coor_image1, sigma_coor_image1, coor_image2, sigma_coor_image2) # in pixel
    print("Sepration =", separation , "+-", sigma_separation, "pixels")
    separation = Pixel_2_distance(separation) # in radians
    sigma_separation = Pixel_2_distance(sigma_separation) # in radians

    print("Separation =", separation, "+-", sigma_separation, "radians")
    thetaE = separation/2
    sigma_thetaE = sigma_separation/2
    velocity_disp, sigma_velocity_disp = Inverted_Einstein_radius(thetaE, sigma_thetaE)
    print("Velocity disperion in unit of c = ", velocity_disp, "+-", sigma_velocity_disp)

    crit_surface_mass_density = c**2/(4*np.pi*G) * D_s / D_d / D_ds
    print("Critical surface mass density = ", crit_surface_mass_density)

    mass_inside_thetaE = np.pi * thetaE**2 * D_d**2 * crit_surface_mass_density
    sigma_mass_inside_thetaE = 2 * mass_inside_thetaE / thetaE * sigma_thetaE
    print("Mass inside Einstein radius = ", mass_inside_thetaE, "+-", sigma_mass_inside_thetaE, "kg")
    mass_inside_thetaE /= 1.9891e30
    sigma_mass_inside_thetaE /= 1.9891e30
    print("Mass inside Einstein radius = %e +- %e solar masses" % (mass_inside_thetaE, sigma_mass_inside_thetaE))

