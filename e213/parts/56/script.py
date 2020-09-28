import numpy as np
import matplotlib.pyplot as plt
import os

directory = "./"
for filename in os.listdir(directory):
    if filename.endswith(".dat"):
        print("Reading ", filename)
        data = np.genfromtxt(filename, delimiter=";", skip_header=1).T
        # genfromtxt cannot read strings, ie the comments
        event = data[0]
        Ctrk_N = data[1]
        Ctrk_Sump = data[2]
        Ecal = data[3]
        Hcal = data[4]

        k = 0
        plt.figure(num=k, figsize=(10, 7))
        plt.hist(Ctrk_N, bins=[-0.5, 0.5, 1.5, 2.5, 3.5])
        plt.xlabel("number of charged tracks")
        plt.savefig(filename.replace(".dat", "") + "_Ctrk_N.pdf",
                    bbox_inches="tight")
        plt.close(k)
        k += 1

        plt.figure(num=k, figsize=(10, 7))
        plt.hist(Ctrk_Sump, bins=5)
        plt.xlabel("sum of momenta")
        plt.savefig(filename.replace(".dat", "") + "_Ctrl_Sump.pdf",
                    bbox_inches="tight")
        plt.close(k)
        k += 1

        plt.figure(num=k, figsize=(10, 7))
        plt.hist(Ecal, bins=5)
        plt.xlabel("Ecal")
        plt.savefig(filename.replace(".dat", "") + "_Ecal.pdf",
                    bbox_inches="tight")
        plt.close(k)
        k += 1

        plt.figure(num=k, figsize=(10, 7))
        plt.hist(Hcal, bins=5)
        plt.xlabel("Hcal")
        plt.savefig(filename.replace(".dat", "") + "_Hcal.pdf",
                    bbox_inches="tight")
        plt.close(k)
        k += 1
