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
    num = len(profs)
    col = palette1(0.9*(i/num) + 0.05)
    mass = str(round(profs[i].star_mass, 3))
    year = str(round(profs[i].star_age, 3))
    model = str(round(profs[i].model_number, 3))
    lab = year + " yr, " + mass + " $M_{\\odot}$, " + model
    plt.plot(profs[i].radius, profs[i].x_mass_fraction_H,
             ls = '-', linewidth = 2, color=col, label=lab)

plt.title("Ilie 4 (100 $m_{\odot}$, Windhorst Inlists)")
plt.legend()
plt.xlabel('Radius [$R_{\odot}$]')
# plt.ticklabel_format(axis="x", style="plain")
# plt.ticklabel_format(axis="y", style="plain")
plt.ylabel('Hydrogen Fraction X')
plt.savefig(str("H_profile_"+ "4" + ".png"), dpi = 400)
# plt.savefig(str("He_profile_"+ ttl + ".pdf"))
plt.clf()


# T
for i in range(len(profs)):
    # set label
    num = len(profs)
    col = palette1(0.9*(i/num) + 0.05)
    mass = str(round(profs[i].star_mass, 3))
    year = str(round(profs[i].star_age, 3))
    model = str(round(profs[i].model_number, 3))
    lab = year + " yr, " + mass + " $M_{\\odot}$, " + model
    plt.plot(profs[i].radius, profs[i].temperature,
             ls = '-', linewidth = 2, color=col, label=lab)

plt.title("Ilie 4 (100 $m_{\odot}$, Windhorst Inlists)")
plt.legend()
plt.xlabel('Radius [$R_{\odot}$]')
# plt.ticklabel_format(axis="x", style="plain")
# plt.ticklabel_format(axis="y", style="plain")
plt.ylabel('Temperature [K]')
plt.yscale("log")
plt.savefig(str("T_profile_"+ "4" + ".png"), dpi = 400)
plt.clf()


# rho
for i in range(len(profs)):
    # set label
    num = len(profs)
    col = palette1(0.9*(i/num) + 0.05)
    mass = str(round(profs[i].star_mass, 3))
    year = str(round(profs[i].star_age, 3))
    model = str(round(profs[i].model_number, 3))
    lab = year + " yr, " + mass + " $M_{\\odot}$, " + model
    plt.plot(profs[i].radius, profs[i].grav,
             ls = '-', linewidth = 2, color=col, label=lab)

plt.title("Ilie 4 (100 $m_{\odot}$, Windhorst Inlists)")
plt.legend()
plt.xlabel('Radius [$R_{\odot}$]')
# plt.ticklabel_format(axis="x", style="plain")
# plt.ticklabel_format(axis="y", style="plain")
plt.yscale("log")
plt.ylabel('Grav. Acc. [cm/s$^2$]')
plt.savefig(str("g_profile_"+ "4" + ".png"), dpi = 400)
plt.clf()


# rho
for i in range(len(profs)):
    # set label
    num = len(profs)
    col = palette1(0.9*(i/num) + 0.05)
    mass = str(round(profs[i].star_mass, 3))
    year = str(round(profs[i].star_age, 3))
    model = str(round(profs[i].model_number, 3))
    lab = year + " yr, " + mass + " $M_{\\odot}$, " + model
    plt.plot(profs[i].radius, profs[i].rho,
             ls = '-', linewidth = 2, color=col, label=lab)

plt.title("Ilie 4 (100 $m_{\odot}$, Windhorst Inlists)")
plt.legend()
plt.xlabel('Radius [$R_{\odot}$]')
# plt.ticklabel_format(axis="x", style="plain")
# plt.ticklabel_format(axis="y", style="plain")
plt.ylabel('Density [g/cm$^3$]')
plt.yscale("log")
plt.savefig(str("rho_profile_"+ "4" + ".png"), dpi = 400)
plt.clf()



# He H
for i in range(len(profs)):
    # set label
    num = len(profs)
    col = palette1(0.9*(i/num) + 0.05)
    mass = str(round(profs[i].star_mass, 3))
    year = str(round(profs[i].star_age, 3))
    model = str(round(profs[i].model_number, 3))
    lab = year + " yr, " + mass + " $M_{\\odot}$, " + model
    Rstar = profs[i].photosphere_r
    plt.plot(profs[i].radius/Rstar, profs[i].x_mass_fraction_H,
             ls = '-', linewidth = 2, color=col, label=lab)

plt.title("Ilie 4 (100 $m_{\odot}$, Windhorst Inlists)")
plt.legend()
plt.xlabel('Fractional Radius $r/R_{*}$')
# plt.ticklabel_format(axis="x", style="plain")
# plt.ticklabel_format(axis="y", style="plain")
plt.ylabel('Hydrogen Fraction X')
plt.savefig(str("H_profile_r_"+ "4" + ".png"), dpi = 400)
# plt.savefig(str("He_profile_"+ ttl + ".pdf"))
plt.clf()


# T
for i in range(len(profs)):
    # set label
    num = len(profs)
    col = palette1(0.9*(i/num) + 0.05)
    mass = str(round(profs[i].star_mass, 3))
    year = str(round(profs[i].star_age, 3))
    model = str(round(profs[i].model_number, 3))
    lab = year + " yr, " + mass + " $M_{\\odot}$, " + model
    Rstar = profs[i].photosphere_r
    plt.plot(profs[i].radius/Rstar, profs[i].temperature,
             ls = '-', linewidth = 2, color=col, label=lab)

plt.title("Ilie 4 (100 $m_{\odot}$, Windhorst Inlists)")
plt.legend()
plt.xlabel('Fractional Radius $r/R_{*}$')
# plt.ticklabel_format(axis="x", style="plain")
# plt.ticklabel_format(axis="y", style="plain")
plt.ylabel('Temperature [K]')
plt.yscale("log")
plt.savefig(str("T_profile_r_"+ "4" + ".png"), dpi = 400)
plt.clf()


# g
for i in range(len(profs)):
    # set label
    num = len(profs)
    col = palette1(0.9*(i/num) + 0.05)
    mass = str(round(profs[i].star_mass, 3))
    year = str(round(profs[i].star_age, 3))
    model = str(round(profs[i].model_number, 3))
    lab = year + " yr, " + mass + " $M_{\\odot}$, " + model
    Rstar = profs[i].photosphere_r
    plt.plot(profs[i].radius/Rstar, profs[i].grav,
             ls = '-', linewidth = 2, color=col, label=lab)

plt.title("Ilie 4 (100 $m_{\odot}$, Windhorst Inlists)")
plt.legend()
plt.xlabel('Fractional Radius $r/R_{*}$')
# plt.ticklabel_format(axis="x", style="plain")
# plt.ticklabel_format(axis="y", style="plain")
plt.yscale("log")
plt.ylabel('Grav. Acc. [cm/s$^2$]')
plt.savefig(str("g_profile_r_"+ "4" + ".png"), dpi = 400)
plt.clf()


# rho
for i in range(len(profs)):
    # set label
    num = len(profs)
    col = palette1(0.9*(i/num) + 0.05)
    mass = str(round(profs[i].star_mass, 3))
    year = str(round(profs[i].star_age, 3))
    model = str(round(profs[i].model_number, 3))
    lab = year + " yr, " + mass + " $M_{\\odot}$, " + model
    Rstar = profs[i].photosphere_r
    plt.plot(profs[i].radius/Rstar, profs[i].rho,
             ls = '-', linewidth = 2, color=col, label=lab)

plt.title("Ilie 4 (100 $m_{\odot}$, Windhorst Inlists)")
plt.legend()
plt.xlabel('Fractional Radius $r/R_{*}$')
# plt.ticklabel_format(axis="x", style="plain")
# plt.ticklabel_format(axis="y", style="plain")
plt.ylabel('Density [g/cm$^3$]')
plt.yscale("log")
plt.savefig(str("rho_profile_r_"+ "4" + ".png"), dpi = 400)
plt.clf()

