#!/usr/bin/env python
# this script reads MESA radial profile data files
# calculates DM temperature acording in Eq. 4.10 in SP85 (Spergel and Press, 1985)
# then uses that DM temp and data from MESA to calculate DM evaporation rate

####################
# IMPORT LIBRARIES #
####################
import mesa_reader as mr
import sys
import argparse
import numpy as np
import mpmath as mp
import math
from decimal import Decimal as D

import scipy.special as sc
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.interpolate import interpolate
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt
import matplotlib

import time
from IPython.display import clear_output
import csv
import copy


##################
# DEFINE CLASSES #
##################

####################
# DEFINE FUNCTIONS #
####################
def interp(x, y):
    '''takes two mesa data arrays and fits an interoplation function'''
    fit = interpolate.interp1d(x, y, fill_value="extrapolate")
    return fit

########
# MAIN #
########
def main():
    # parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--direc", help="directory containing MESA profile and history files")
    parser.add_argument("-p", "--profile", help="index of the profile to use", type=int)
    parser.add_argument("-T", "--TchiMchi", help="plot DM temperature vs DM mass", action='store_true')
    parser.add_argument("-t", "--taumu", help="plot DM dimensionless temperature vs DM dimensionless mass", action='store_true')
    parser.add_argument("-V", "--phi", help="plot radial graviation potential from MESA data files", action='store_true')
    parser.add_argument("-v", "--phipoly", help="plot radial graviation potential for N=3 polytrope", action='store_true')
    parser.add_argument("-n", "--np", help="plot proton number denisty from MESA data files", action='store_true')
    parser.add_argument("-e", "--evap", help="plot DM evap rate from MESA data files", action='store_true')

    # arguments
    args = parser.parse_args()

    # ploting stuff
    fig = plt.figure(figsize = (12,8))
    plt.style.use('fast')
    palette = plt.get_cmap('viridis')

    # read in MESA data from files specified in command line arguments
    if args.direc and args.profile:
        arg1 = args.direc
        arg = arg1.split('_')
        hist= "history_" + arg[0] + ".data"
        direc = mr.MesaLogDir(log_path=arg1, history_file=hist)
        prof = direc.profile_data(int(args.profile))

        # read info about the MESA star
        mass = str(round(prof.star_mass, 3))
        year = str(round(prof.star_age, 3))
        model = str(round(prof.model_number, 3))
        mesa_lab = year + " yr, " + mass + " $M_{\\odot}$, " + model

        # interpolate temp wrt radius
        # T = interp(prof.radius, prof.mass)
        # T = interp(prof.radius, prof.pressure)
        # T = interp(prof.radius, prof.grav)
        # T = interp(prof.radius, prof.rho)
        T = interp(prof.radius, prof.y_mass_fraction_He)
        r = np.linspace(0.0, 5.0, num=10000)

    # plt.scatter(prof.radius, prof.mass, ls = '-', linewidth = 1, label=mesa_lab)
    # plt.scatter(prof.radius, prof.pressure, ls = '-', linewidth = 1, label=mesa_lab)
    # plt.scatter(prof.radius, prof.grav, ls = '-', linewidth = 1, label=mesa_lab)
    # plt.scatter(prof.radius, prof.rho, ls = '-', linewidth = 1, label=mesa_lab)
    plt.scatter(prof.radius, prof.y_mass_fraction_He, ls = '-', linewidth = 1, label=mesa_lab)
    plt.plot(r, T(r), ls = '-', linewidth = 1, label="fit")
    plt.title("MESA DM Temperature $100 M_{\\odot}$ (Windhorst)")
    plt.legend()
    plt.xlabel('$m_{\\chi}$ [Gev]')
    plt.ylabel('$T_{\\chi}$ [K]')
    # plt.yscale("log")
    # plt.xscale("log")
    plt.show()
    plt.clf()

###########
# EXECUTE #
###########
if __name__ == "__main__":
    # execute only if run as a script
    main()
