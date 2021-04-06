#!/usr/bin/env python
import mesa_reader as mr
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys


# set up some ploting stuff
fig = plt.figure(figsize = (12,8))
plt.style.use('fast')
palette = plt.get_cmap('magma')
palette1 = plt.get_cmap('viridis')


# command line arguments
ddd = str(sys.argv[1])
yyy = ddd.split('_')
hhh = "history_" + yyy[0] + ".data"
direc = mr.MesaLogDir(log_path=ddd, history_file=hhh)
profs = []
names = []
for i in range(2, len(sys.argv)):
    profs.append(direc.profile_data(int(sys.argv[i])))
    names.append("model " + str(sys.argv[i]))
ttl = ddd[:2]


# He H
for i in range(len(profs)):
    # set label
    mass = str(round(profs[i].star_mass, 3))
    year = str(round(profs[i].star_age, 3))
    model = str(round(profs[i].model_number, 3))
    lab = year + " yr, " + mass + " $M_{\\odot}$, " + model
    plt.plot(profs[i].radius, profs[i].y_mass_fraction_He, ls = '-', linewidth = 2, color=palette(.5), label=lab)

plt.title("Helium Fraction in Pop. III Stars")
plt.legend()
plt.xlabel('Radius [$R_{\odot}$]')
plt.ticklabel_format(axis="x", style="plain")
plt.ticklabel_format(axis="y", style="plain")
plt.ylabel('Helium Fraction')
plt.savefig(str("He_profile_"+ ttl + ".png"), dpi = 400)
# plt.savefig(str("He_profile_"+ ttl + ".pdf"))
plt.clf()




