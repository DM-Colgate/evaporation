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
def SP85_EQ410(T_chi):
    ''' uses MESA data in arrays, phi_mesa, n_mesa, prof.temperature, prof.radius_cm to evalute the integral in EQ. 4.10 from SP85'''
    # this is to fixed weird array inside an array stuff I don't really understand
    diff = []
    for k in range(len(prof.radius_cm)):
        diff.append(prof.temperature[k] - T_chi)
    diff = np.concatenate(diff, axis=0)

    # the integrand from EQ. 4.10
    integrand_410 = []
    for k in range(len(prof.radius_cm)):
        integrand_410.append(n_mesa[k] * math.sqrt((m_p* T_chi + m_chi * prof.temperature[k])/(m_chi*m_p)) * diff[k] * math.exp((-1*m_chi*phi_mesa[k])/(k_cgs*T_chi)) * prof.radius_cm[k]**2)
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

def calc_n_mesa(prof):
    ''' calculate proton number density using rho given by mesa'''
    n_mesa = []
    for k in range(len(prof.rho)):
        n_mesa.append(prof.x_mass_fraction_H[k] * prof.rho[k])
        n_mesa[k] = n_mesa[k] / m_p_cgs
    return n_mesa

def interp(x, y):
    '''takes two mesa data arrays and fits an interoplation function'''
    fit = interpolate.interp1d(x, y, fill_value="extrapolate")
    return fit

def f_chi(w, r):
    term1 = np.exp(-1 * w**2 / v_chi(r)**2)
    #TODO: second argument?
    term2 = np.heaviside(v_c(r) - w, 0.5)
    term3 = np.sqrt(np.pi**3) * v_chi(r)**3
    term4 = sc.erf(v_c(r)/v_chi(r))
    term5 = 2/np.pi * (v_c(r))/(v_chi(r)) * np.exp( sc.erf((v_c(r)**2)/(v_chi(r)**2)) )
    return (term1*term2)/(term3*(term4 - term5))

def v_chi(r):
    return np.sqrt(2*T_chi(r)/m_chi)

def v_c(r):
    return vesc(r)

def R(pm, )
    if (plus_minus == '+'):

    elif(plus_minus == '-'):

def u(r):
    return(2 * T(r)/m_p_cgs)

def alpha(pm, r, w, v):
    if (plus_minus == '+'):
        val = u(r) * (mu_plus(mu(mchi)) * v + mu_minus(mu(mchi)) * w)
    if (plus_minus == '-'):
        val = u(r) * (mu_plus(mu(mchi)) * v - mu_minus(mu(mchi)) * w)
    return val

def beta(pm):
    if (plus_minus == '+'):
        val = u(r) * (mu_minus(mu(mchi)) * v + mu_plus(mu(mchi)) * w)
    if (plus_minus == '-'):
        val = u(r) * (mu_minus(mu(mchi)) * v - mu_plus(mu(mchi)) * w)
    return val

def chi():

def mu(m_chi):
   return g_per_GeV * m_chi / m_p

def mu_plus(mu):
   return mu + 1 / 2

def mu_minus(mu):
   return mu - 1 / 2

def vesc(r):
    return sqrt(2*(G_cgs * M_star_cgs * R_star_cgs + (phi(R_star) - phi(r)))

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

    # globals
    global prof
    global m_p_cgs
    global k_cgs
    global g_per_GeV
    global G_cgs
    global g_per_Msun
    global phi_mesa
    global n_mesa
    global T
    global n
    global phi
    global m_chi
    global T_chi
    m_p_cgs = 1.6726231 * 10 ** (-24) # grams
    k_cgs = 1.3807 * 10 ** (-16) # cm2 g s-2 K-1 
    g_per_GeV = 5.61 *10 ** (-23)
    G_cgs = 6.6743*10**(-8) #cgs
    g_per_Msun = 1.988*10**33


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

        # calculate phi and number density
        n_mesa = calc_n_mesa(prof)
        phi_mesa = calc_phi_mesa(prof)

        # calculate DM temp
        # masses to test
        m_chi_sample = [0.00001, 0.000015, 0.00002, 0.00003, 0.00005, 0.00007, 0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000, 15000]

        # use central temp to guess
        T_chi_guess = prof.temperature[-1]
        T_chi_sample = []

        # do MESA calcs and run thru all massses
        for i in range(len(m_chi_sample)):
            m_chi = g_per_GeV * m_chi_sample[i]
            print("Solving Tchi for Mchi =", m_chi_sample[i], "Gev...")
            # numerically solve
            T_chi_sample.append(fsolve(SP85_EQ410, T_chi_guess))

        # now fit interpolation functions to T, n and phi wrt radius 
        T = interp(prof.radius_cm, prof.temperature)
        n = interp(prof.radius_cm, n_mesa)
        phi = interp(prof.radius_cm, phi_mesa)

        # interp wrt DM mass
        T_chi = interp(m_chi_sample, T_chi_sample)

        # interpolate temp wrt radius
        # T = interp(prof.radius, prof.mass)
        # T = interp(prof.radius, prof.pressure)
        # T = interp(prof.radius, prof.grav)
        # T = interp(prof.radius, prof.rho)
        # T = interp(prof.radius, prof.y_mass_fraction_He)
        # r = np.linspace(0.0, 5.0, num=10000)

    # plt.scatter(prof.radius, prof.mass, ls = '-', linewidth = 1, label=mesa_lab)
    # plt.scatter(prof.radius, prof.pressure, ls = '-', linewidth = 1, label=mesa_lab)
    # plt.scatter(prof.radius, prof.grav, ls = '-', linewidth = 1, label=mesa_lab)
    # plt.scatter(prof.radius, prof.rho, ls = '-', linewidth = 1, label=mesa_lab)
    # plt.scatter(prof.radius, prof.y_mass_fraction_He, ls = '-', linewidth = 1, label=mesa_lab)
    # plt.plot(r, T(r), ls = '-', linewidth = 1, label="fit")
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
