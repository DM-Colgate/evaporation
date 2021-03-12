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
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt
import matplotlib

import time
from IPython.display import clear_output
import csv
import copy


####################
# DEFINE FUNCTIONS #
####################
def SP85_EQ410(Tchi):
    ''' uses MESA data in arrays, phi_mesa, np_mesa, T_mesa, r_mesa to evalute the integral in EQ. 4.10 from SP85'''
    # this is to fixed weird array inside an array stuff I don't really understand
    diff = []
    for k in range(len(r_mesa)):
        diff.append(T_mesa[k] - Tchi)
    diff = np.concatenate(diff, axis=0)

    # the integrand from EQ. 4.10
    integrand_410 = []
    for k in range(len(r_mesa)):
        integrand_410.append(np_mesa[k] * math.sqrt((m_p* Tchi + mchi * T_mesa[k])/(mchi*m_p)) * diff[k] * math.exp((-1*mchi*phi_mesa[k])/(k_cgs*Tchi)) * r_mesa[k]**2)
    return np.trapz(integrand_410, x=r_mesa)

def calc_phi_mesa(prof):
    ''' calculate potential from accleration given by mesa'''
    phi = []
    r = []
    acc = []
    for k in range(len(prof.grav)):
        # create an array of raddii and phis only interior of our point i
        # in MESA 1st cell is the surface, last cell is the center
        # \/ lists that exclude cells exterior to i \/
        r = prof.radius_cm[k:]
        acc = prof.grav[k:]
        # integate over the grav. acc. w.r.t. radius up to the point i
        phi.append(np.trapz(-1*acc, x=r))
    return phi

def calc_np_mesa(prof):
    ''' calculate proton number density using rho given by mesa'''
    np_mesa = []
    for k in range(len(prof.rho)):
        np_mesa.append(prof.x_mass_fraction_H[k] * prof.rho[k])
        np_mesa[k] = np_mesa[k] / m_p
    return np_mesa

def calc_r_mesa_frac(prof):
    ''' calculate dimesnionless radius'''
    r_mesa_frac = []
    for k in range(len(prof.radius)):
        r_mesa_frac.append(prof.radius[k] / prof.radius[0])
    return r_mesa_frac

def calc_Tchi(func, Tchi_guess):
    ''' returns for what Tchi the input function is zero'''
    return fsolve(func, Tchi_guess)


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

    args = parser.parse_args()

    # set variables
    global mchi
    global k_cgs
    global g_per_GeV
    global G_cgs
    global m_p
    global xis
    global xis_frac
    global theta
    mchi = 1.6726231 * 10 ** (-24) # grams
    m_p = 1.6726231 * 10 ** (-24) # grams
    k_cgs = 1.3807 * 10 ** (-16) # cm2 g s-2 K-1 
    g_per_GeV = 5.61 *10 ** (-23)
    G_cgs = 6.6743*10**(-8) #cgs
    g_per_Msun = 1.988*10**33

    # read in MESA data from files specified in command line arguments
    if args.direc and args.profile:
        arg1 = args.direc
        arg = arg1.split('_')
        hist= "history_" + arg[0] + ".data"
        direc = mr.MesaLogDir(log_path=arg1, history_file=hist)
        prof = direc.profile_data(int(args.profile))

        # calculate the gravitational potential
        global phi_mesa
        phi_mesa = calc_phi_mesa(prof)

        # calculate the proton number density
        global np_mesa
        np_mesa = calc_np_mesa(prof)

        # set temp and radius
        global T_mesa
        global r_mesa
        global r_mesa_cgs
        global r_mesa_frac
        T_mesa = prof.temperature
        r_mesa = prof.radius
        r_mesa_cgs = prof.radius_cm
        r_mesa_frac = calc_r_mesa_frac(prof)

        # read info about the MESA star
        mass = str(round(prof.star_mass, 3))
        year = str(round(prof.star_age, 3))
        model = str(round(prof.model_number, 3))
        mesa_lab = year + " yr, " + mass + " $M_{\\odot}$, " + model

        # use central temp to guess
        Tchi_guess = T_mesa[-1]

        # masses to test
        mchi_sample = [0.00001, 0.000015, 0.00002, 0.00003, 0.00005, 0.00007, 0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000, 15000]
        Tchi_sample = []

        # do MESA calcs and run thru all massses
        for i in range(len(mchi_sample)):
            mchi = g_per_GeV * mchi_sample[i]
            print("Solving Tchi for Mchi =", mchi_sample[i], "Gev...")
            # numerically solve
            Tchi_sample.append(calc_Tchi(SP85_EQ410, Tchi_guess))

        # convert to dimensionless
        mu_sample = []
        Tau_sample = []
        for i in range(len(mchi_sample)):
            mu_sample.append(g_per_GeV * mchi_sample[i] / m_p)
            Tau_sample.append(Tchi_sample[i] / T_mesa[-1])

    #Taking typical sigma value
    sigma = 1e-43

    #DM Mass
    mx = np.logspace(-4, 0, 30)

    # plot formatting
    fig = plt.figure(figsize = (12,8))
    plt.style.use('fast')
    palette = plt.get_cmap('viridis')

    # write to CSV
    m_chi_sample = np.asarray(mchi_sample)
    T_chi_sample = np.asarray(Tchi_sample)
    output = np.column_stack((m_chi_sample.flatten(), T_chi_sample.flatten()))
    np.savetxt('TM4.csv',output,delimiter=',')


    # plot Tchi vs. Mchi
    if args.TchiMchi == True:
        plt.plot(mchi_sample, Tchi_sample, ls = '-', linewidth = 1, label=mesa_lab)
        plt.title("MESA DM Temperature $100 M_{\\odot}$ (Windhorst)")
        plt.legend()
        plt.xlabel('$m_{\\chi}$ [Gev]')
        plt.ylabel('$T_{\\chi}$ [K]')
        # plt.yscale("log")
        plt.xscale("log")
        plt.show()
        plt.clf()

    # plot tau vs. mu
    if args.taumu == True:
        plt.plot(mu_sample, Tau_sample, ls = '-', linewidth = 1, label=mesa_lab)
        plt.plot(mx_tau_fit, tau_temp, ls = '-', linewidth = 1, label="$100 M_{\odot}$ N=3")
        plt.title("MESA DM Temperature $100 M_{\\odot}$ (Windhorst)")
        plt.legend()
        plt.xlabel('$ \\mu $')
        plt.ylabel('$ \\tau $')
        # plt.yscale("log")
        plt.xscale("log")
        plt.show()
        plt.clf()

    # plot phi vs. r
    if args.phipoly == True or args.phi == True:
        if args.phi == True:
            plt.plot(r_mesa_frac, phi_mesa, ls = '-', linewidth = 1, label=mesa_lab)
        if args.phipoly == True:
            plt.plot(xis_frac, phi_xi_poly, ls = '-', linewidth = 1, label="$100 M_{\\odot}$ N=3")
        plt.title("Grav. Acc. (Windhorst and Polytrope)")
        plt.legend()
        plt.xlabel('$ r / R_{*} $')
        plt.ylabel('$ \\phi (r) \\; [cm^{2} \\cdot s^{-2}] $')
        # plt.yscale("log")
        # plt.xscale("log")
        plt.show()
        plt.clf()


    # evap vs mchi
    if args.evap == True:
        plt.plot(mx, E_G, label = 'Numerical, $M_\star = %i$'%star.mass, ls = '-')
        plt.plot(mx, E_ilie2, label = 'Approximate Solution (New)', ls = '--')
        plt.plot(mx, E_ilie, label = 'Approximate Solution (Old)', ls = '-.')
        plt.plot(mx, E_mesa, label = mesa_lab, ls = '-.')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$m_X$ [GeV]', fontsize = 15)
        plt.ylabel('$E$ [s$^{-1}$]', fontsize = 15)
        plt.xlim(mx[0], mx[-1])
        plt.title('DM Evaporation Rate in Population III Stars, $\\sigma = 10^{%i}$'%np.log10(sigma), fontsize = 15)
        plt.legend(loc = 'best')
        plt.savefig('Evap_mx_NumvsApprox.pdf', dpi = 200, bbox_inches = 'tight', pad_inches = 0)
        plt.show()


###########
# EXECUTE #
###########
if __name__ == "__main__":
    # execute only if run as a script
    main()


