import numpy as np
import matplotlib.pyplot as plt

G_F = 1.1663e-5  # GeV-2
sin2_thetaW = 0.2312
thetaW = np.arcsin(np.sqrt(sin2_thetaW))
cos2_thetaW = 1 - sin2_thetaW
M_Z = 91.182  # GeV

e_dict = {
    "name": "electron",
    "Q": -1,    # elec. charge
    "Nc": 1,    # color charge
    "LI3": -1/2,
    "RI3": 0,
}

nu_dict = {
    "name": "neutrino",
    "Q": 0,
    "Nc": 1,
    "LI3": 1/2,
}

up_q_dict = {
    "name": "up quarks",
    "Q": 2/3,
    "Nc": 3,
    "LI3": 1/2,
    "RI3": 0,
}


down_q_dict = {
    "name": "down quarks",
    "Q": -1/3,
    "Nc": 3,
    "LI3": -1/2,
    "RI3": 0,
}


def vector_cplg(I3, Qf):
    return I3 - 2 * Qf * sin2_thetaW


def axial_vector_cplg(I3):
    return I3


def Gamma_f(f_dict):
    Gamma = (vector_cplg(f_dict["LI3"], f_dict["Q"])**2 + axial_vector_cplg(f_dict["LI3"])**2)

    Gamma *= np.sqrt(2) * f_dict["Nc"] / (12*np.pi) * G_F * M_Z**3
    Gamma *= 1000   # MeV

    return Gamma

def sigma_f(f_dict, s):
    sig = 12 * np.pi / M_Z**2 * s * Gamma_f(e_dict) * Gamma_f(f_dict)
    sig /= ((s-M_Z**2)**2 + s**2 * Gamma_Z**2 / M_Z**2)
    return sig


def s_diff_cross(cos_theta):
    return 1 + cos_theta**2


def t_diff_cross(cos_theta):
    return (1-cos_theta)**(-2)


def total_diff_cross(theta):
    return s_diff_cross(theta) + t_diff_cross(theta)


def v_f(f_dict):
    return vector_cplg(f_dict["LI3"], f_dict["Q"])/(2*np.cos(thetaW)*np.sin(thetaW))


def a_f(f_dict):
    return axial_vector_cplg(f_dict["LI3"])/(2*np.cos(thetaW)*np.sin(thetaW))


def fb_asymmetry(f_dict, s):
    A_FB = -3/2
    A_FB *= a_f(e_dict) * a_f(f_dict) * f_dict["Q"]
    A_FB /= ((v_f(e_dict)**2 + a_f(e_dict)**2) * (v_f(f_dict)**2 + a_f(f_dict)**2))
    A_FB *= s * (s - M_Z**2)
    A_FB /= ((s-M_Z**2)**2 + (s*Gamma_Z/M_Z)**2)
    return A_FB


if __name__ == "__main__":
    # Q5.1
    print("Decay width to", e_dict["name"], "=", Gamma_f(e_dict), "MeV")
    print("Decay width to", nu_dict["name"], "=", Gamma_f(nu_dict), "MeV")

    print("Decay width to", up_q_dict["name"], "=", Gamma_f(up_q_dict), "MeV")
    print("Decay width to", down_q_dict["name"], "=", Gamma_f(down_q_dict), "MeV")

    # Q5.2
    Gamma_e = Gamma_f(e_dict)
    Gamma_ch = 3*Gamma_e
    print("Decay width to charged particles =", Gamma_ch, "MeV")
    Gamma_nu = Gamma_f(nu_dict)
    Gamma_inv = 3*Gamma_nu
    print("Decay width to invisible particles =", Gamma_inv, "MeV")
    Gamma_up = Gamma_f(up_q_dict)
    Gamma_down = Gamma_f(down_q_dict)
    Gamma_q = 2*Gamma_up + 3*Gamma_down
    print("Decay width to q =", Gamma_q, "MeV")
    Gamma_Z = Gamma_ch + Gamma_inv + Gamma_q
    print("Total decay width =", Gamma_Z, "MeV")

    sigma_ch = 3 * sigma_f(e_dict, M_Z**2)
    print("Partial cross section to charged particles =", sigma_ch, "MeV-2")
    sigma_inv = 3 * sigma_f(nu_dict, M_Z**2)
    print("Partial cross section to invisibale particles =", sigma_inv, "MeV-2")
    sigma_q = 2 * sigma_f(up_q_dict, M_Z**2) + 3 * sigma_f(down_q_dict, M_Z**2)
    print("Partial cross section to hardrons =", sigma_q, "MeV-2")
    sigma_total = sigma_ch + sigma_inv + sigma_q
    print("Total cross section =", sigma_total, "MeV-2")

    # Q5.3
    Gamma_prime_Z = Gamma_Z + Gamma_e + Gamma_nu + Gamma_up + Gamma_down
    change = Gamma_prime_Z - Gamma_Z
    change /= Gamma_Z
    print("Total width with additional generation =", Gamma_prime_Z, "MeV",
          "(a change of ", change, "percent)")

    # Q5.4
    x = np.linspace(-0.9, 0.8, 100)
    k = 0
    plt.figure(num=k, figsize=(10,7))
    plt.plot(x, s_diff_cross(x), label="s-channel", color="blue")
    plt.plot(x, t_diff_cross(x), label="t-channel", color="red")
    plt.plot(x, total_diff_cross(x), label="total", color="black")
    plt.legend()
    plt.xlabel("$\Theta$")
    plt.ylabel(r"$\sim \frac{d \sigma}{d \Omega}$")
    plt.savefig("./angDep.pdf", bbox_inches="tight")
    plt.close(k)
    k += 1

    # Q5.5
    for s in [89.225, 91.225, 93.225]:
        for sin2_thetaW in [0.21, 0.23, 0.25]:
            thetaW = np.arcsin(np.sqrt(sin2_thetaW))
            print("Energy:", s, "sin2_thetaW:", sin2_thetaW, "Asymmetry:", fb_asymmetry(e_dict, s))

    # reset
    sin2_thetaW = 0.2312
thetaW = np.arcsin(np.sqrt(sin2_thetaW))
