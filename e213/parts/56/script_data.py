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
        plt.figure(num=k, figsize=(7, 5))
        plt.hist(Ctrk_N, bins=5)
        plt.xlabel("Number of charged tracks")
        plt.ylabel("Number of events")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig(filename.replace(".dat", "") + "_Ctrk_N.pdf", bbox_inches="tight")
        plt.close(k)
        k += 1

        plt.figure(num=k, figsize=(7, 5))
        plt.hist(Ctrk_Sump, bins=8)
        plt.xlabel("Sum of momenta [GeV]")
        plt.ylabel("Number of events")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig(filename.replace(".dat", "") + "_Ctrl_Sump.pdf", bbox_inches="tight")
        plt.close(k)
        k += 1

        plt.figure(num=k, figsize=(7, 5))
        plt.hist(Ecal, bins=8)
        plt.xlabel("Ecal [GeV]")
        plt.ylabel("Number of events")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig(filename.replace(".dat", "") + "_Ecal.pdf", bbox_inches="tight")
        plt.close(k)
        k += 1

        plt.figure(num=k, figsize=(7, 5))
        plt.hist(Hcal, bins=8)
        plt.xlabel("Hcal [GeV]")
        plt.ylabel("Number of events")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig(filename.replace(".dat", "") + "_Hcal.pdf", bbox_inches="tight")
        plt.close(k)
        k += 1
