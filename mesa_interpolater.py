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

def R311(r, T_chi, m_chi, sigma):
    '''Eq. 3.11 from Goulde 1987, normalized evap. rate'''
    #TODO numerical integration over phase space
    a1 = (2/np.pi)*(2*T(r)/m_chi)**(1/2)
    a2 = (T(r)/T_chi**(3/2))* sigma * n_p(r) * n_chi(r, T_chi, m_chi)
    b3 = np.exp(-1*(mu_plus(mu(m_chi))/xi(r, m_chi, T_chi))**2 *(m_chi*v_esc(r)**2/(2*T_chi)))
    c4 = mu(m_chi) * mu_minus(mu(m_chi)) / (T(r)*mu(m_chi)*xi(r, m_chi, T_chi)/T_chi)
    d5 = (xi(r, m_chi, T_chi)**2) / (T(r)*mu(m_chi)/T_chi)
    d6 = mu_plus(mu(m_chi)) * mu_minus(mu(m_chi)) / (mu(m_chi))
    c7 = (mu_plus(mu(m_chi))**3) / (xi(r, m_chi, T_chi) * ( (T(r)*mu(m_chi)/T_chi - mu(m_chi))))
    b8 = chi(gamma('-',  r, m_chi, T_chi), gamma('+',  r, m_chi, T_chi))
    b9 = np.exp(-1* m_chi * v_c(r)**2 / (2*T_chi) * (mu(m_chi)*T_chi) / (T(r)*mu(m_chi)))
    c10 = alpha('-', r, m_chi, v_c(r), v_esc(r)) * alpha('+', r, m_chi, v_c(r), v_esc(r))
    c11 = 1/(2*mu(m_chi))
    c12 = mu_minus(mu(m_chi))**2 *(1/mu(m_chi)) - (T_chi/(T(r)*mu(m_chi)))
    b13 = chi(alpha('+', r, m_chi, v_c(r), v_esc(r)), alpha('-', r, m_chi, v_c(r), v_esc(r)))
    b14 = np.exp(-1* m_chi * v_c(r)**2 / (2*T_chi)) * np.exp(-1*((m_chi*v_esc(r)**2 /2) - (m_chi*v_c(r)**2 /2))/T(r))
    b15 = mu_plus(mu(m_chi))**2 / ((T(r)*mu(m_chi)/T_chi) - mu(m_chi))
    b16 = chi(beta('-', r, m_chi, v_c(r), v_esc(r)), beta('+', r, m_chi, v_c(r), v_esc(r)))
    b17 = np.exp(-1 * (m_chi* v_chi(r, m_chi, T_chi)**2)/(2*T_chi) * alpha('-', r, m_chi, v_c(r), v_esc(r))**2)
    b18 = mu(m_chi) * alpha('+', r, m_chi, v_c(r), v_esc(r)) / (2*T(r)*mu(m_chi)/T_chi)
    b19 = np.exp(-1 * (m_chi* v_chi(r, m_chi, T_chi)**2)/(2*T_chi) * alpha('+', r, m_chi, v_c(r), v_esc(r))**2)
    b20 = mu(m_chi) * alpha('-', r, m_chi, v_c(r), v_esc(r)) / (2*T(r)*mu(m_chi)/T_chi)
    # print("shell scattering rate is = ", a1*a2*(b3*(c4*(d5 - d6) + c7)*b8 + b9*(c10 - c11 + c12)*b13 - b14*b15*b16 - b17*b18 + b19*b20))
    return a1*a2*(b3*(c4*(d5 - d6) + c7)*b8 + b9*(c10 - c11 + c12)*b13 - b14*b15*b16 - b17*b18 + b19*b20)

def Omegaplus37(r, w, T_chi, m_chi, sigma):
    '''Eq. 3.7 from Goulde 1987'''
    a1 = (2*T(r))/(2*m_p*np.pi**(1/2))
    a2 = 1/(mu(m_chi)**2)
    a3 = sigma*n_p(r)/w
    b4 = mu(m_chi)
    c5 = alpha('+', r, m_chi, w, v_esc(r)) * np.exp(-1*(alpha('-', r, m_chi, w, v_esc(r)))**2)
    c6 = alpha('-', r, m_chi, w, v_esc(r)) * np.exp(-1*(alpha('+', r, m_chi, w, v_esc(r)))**2)
    c7 = mu(m_chi)
    c8 = alpha('-', r, m_chi, w, v_esc(r)) * np.exp(-1*(alpha('+', r, m_chi, w, v_esc(r)))**2)
    c9 = 2 * mu(m_chi) * alpha('+', r, m_chi, w, v_esc(r)) * alpha('-', r, m_chi, w, v_esc(r))
    c9 = 2 * mu_plus(mu(m_chi)) * mu_minus(mu(m_chi))
    b10 = chi(alpha('-', r, m_chi, w, v_esc(r)), alpha('+', r, m_chi, w, v_esc(r)))
    b11 = 2 * (mu_plus(mu(m_chi)))**2
    b12 = chi(beta('-', r, m_chi, w, v_esc(r)), beta('+', r, m_chi, w, v_esc(r)))
    b13 = np.exp(-1* m_chi * (v_esc(r)**2 - w**2) / (2*T(r)))
    return a1*a2*a3*(b4*(c5 - c6) + (c7 - c8 - c9)*b10 + b11*b12*b13)

def f_w38(r, w, T_chi, m_chi):
    '''Eq. 3.8 from Goulde 1987'''
    a1 = 4/(np.pi**(1/2))
    a2 = m_chi / ((2*T_chi)**(3/2))
    a3 = n_chi(r, T_chi, m_chi) * w**2
    a4 = np.exp(-1 * m_chi * w**2 /(2* T_chi))
    a5 = np.heaviside(v_c(r) - w, 0.5)
    return a1*a2*a3*a4*a5

def R39_integrand(w, r, T_chi, m_chi, sigma):
    return f_w38(r, w, T_chi, m_chi)*Omegaplus37(r, w, T_chi, m_chi, sigma)

def R39(r, T_chi, m_chi, sigma):
    return quad(R39_integrand, 0, np.inf, args=(r, T_chi, m_chi, sigma), limit=500)[0]

def v_chi(r, m_chi, T_chi):
    return np.sqrt(2*T_chi/m_chi)

def n_chi(r, T_chi, m_chi):
    '''normalized isotropic DM distribution using user supplied potential and DM temp (from MESA)'''
    return np.exp(-1.0*m_chi*phi(r)/(k_cgs*T_chi))

def alpha(pm, r, m_chi, w, v):
    '''made up goulde function'''
    if (pm == '+'):
        val = m_p/(2 * T(r))**(1/2) * (mu_plus(mu(m_chi)) * v + mu_minus(mu(m_chi)) * w)
    if (pm == '-'):
        val = m_p/(2 * T(r))**(1/2) * (mu_plus(mu(m_chi)) * v - mu_minus(mu(m_chi)) * w)
    return val

def beta(pm, r, m_chi, w, v):
    '''made up goulde function'''
    if (pm == '+'):
        val = m_p/(2 * T(r))**(1/2) * (mu_minus(mu(m_chi)) * v + mu_plus(mu(m_chi)) * w)
    if (pm == '-'):
        val = m_p/(2 * T(r))**(1/2) * (mu_minus(mu(m_chi)) * v - mu_plus(mu(m_chi)) * w)
    return val

def gamma(pm, r, m_chi, T_chi):
    '''made up goulde function'''
    if (pm == '+'):
        val = (m_p/(2*T(r)))**(1/2) * ((mu_plus(mu(m_chi)) * mu_minus(mu(m_chi)) )*v_esc(r)/xi(r, m_chi, T_chi) + xi(r, m_chi, T_chi)*v_c(r))
    if (pm == '-'):
        val = (m_p/(2*T(r)))**(1/2) * ((mu_plus(mu(m_chi)) * mu_minus(mu(m_chi)) )*v_esc(r)/xi(r, m_chi, T_chi) - xi(r, m_chi, T_chi)*v_c(r))
    return val

def xi(r, m_chi, T_chi):
    '''not the polytope xi!!!!!!, just a made up goulde function'''
    return np.sqrt(mu_minus(mu(m_chi))**2 + T(r)/(T_chi*mu(m_chi)))

def chi(a, b):
    #TODO: is this right?
    return np.sqrt(np.pi)/2 * (mp.erf(b) - mp.erf(a))

def mu(m_chi):
    '''dimensional less DM mass'''
    return m_chi / m_p

def mu_plus(mu):
    return mu + 1 / 2

def mu_minus(mu):
    return mu - 1 / 2

def v_c(r):
    return v_esc(r)

def v_esc(r):
    return np.sqrt(2*(G_cgs * M_star_cgs * R_star_cgs + (phi(R_star_cgs) - phi(r))))

def T(r):
    return T_fit(r)

def n_p(r):
    ''' calculate proton number density using rho given by mesa'''
    return x_mass_fraction_H_fit(r) * rho_fit(r) / m_p

def phi_integrand(r):
    # TODO: should this be -1???
    return grav_fit(r)

def phi(r):
    ''' calculate potential from accleration given by mesa'''
    return quad(phi_integrand, 0, r, limit=500)[0]

def rho(r):
    return rho_fit(r)

def SP85_EQ410_integrand(r, T_chi, m_chi):
    t1 = n_p(r)
    t2 = math.sqrt(m_p* T_chi + m_chi * T(r)/(m_chi*m_p))
    t3 = (T(r) - T_chi)
    t4 = math.exp((-1*m_chi*phi(r))/(k_cgs*T_chi))
    return t1 * t2 * t3 * t4 * r**2

def SP85_EQ410(T_chi, m_chi, R_star):
    ''' uses MESA data in arrays, phi_mesa, n_mesa, prof.temperature, prof.radius_cm to evalute the integral in EQ. 4.10 from SP85'''
    return quad(SP85_EQ410_integrand, 0, R_star, args=(T_chi, m_chi), limit=500)[0]

def normfactor(r, m_chi, T_chi):
    t1 = mp.erf(v_c(r)/v_chi(r, m_chi, T_chi))
    t2 = (2/np.pi)*(v_c(r)/v_chi(r, m_chi, T_chi))
    t3 = np.exp(-1*v_c(r)**2 /(v_chi(r, m_chi, T_chi)**2))
    return t1 - t2*t3

def evap_rate_integrand(r, T_chi, m_chi, sigma):
    r311 =  R311(r, T_chi, m_chi, sigma) / normfactor(r, m_chi, T_chi)
    r39 =  R39(r, T_chi, m_chi, sigma) / normfactor(r, m_chi, T_chi)
    print("DIFF = ", r39-r311)
    return r39

def evap_rate(T_chi, m_chi, sigma):
    return quad(evap_rate_integrand, 0, R_star_cgs, args=(T_chi, m_chi, sigma), limit=500)[0] * quad(n_chi, 0, R_star_cgs, args=(T_chi, m_chi), limit=500)[0]

########
# MAIN #
########
def main():
    # parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--direc", help="directory containing MESA profile and history files")
    parser.add_argument("-p", "--profile", help="index of the profile to use", type=int)
    # parser.add_argument("-T", "--TchiMchi", help="plot DM temperature vs DM mass", action='store_true')
    # parser.add_argument("-t", "--taumu", help="plot DM dimensionless temperature vs DM dimensionless mass", action='store_true')
    # parser.add_argument("-V", "--phi", help="plot radial graviation potential from MESA data files", action='store_true')
    # parser.add_argument("-v", "--phipoly", help="plot radial graviation potential for N=3 polytrope", action='store_true')
    # parser.add_argument("-n", "--np", help="plot proton number denisty from MESA data files", action='store_true')
    # parser.add_argument("-e", "--evap", help="plot DM evap rate from MESA data files", action='store_true')

    # arguments
    args = parser.parse_args()

    # main variables
    global m_p
    # m_p = 0.9382720813 #GeV
    m_p = 1.6726231 * 10 ** (-24) # grams
    global g_per_GeV
    g_per_GeV = 1.783 *10 ** (-24)
    global G_cgs
    G_cgs = 6.6743*10**(-8) #cgs
    global k_cgs
    k_cgs = 1.3807 * 10 ** (-16) # cm2 g s-2 K-1
    m_p_cgs = 1.6726231 * 10 ** (-24) # grams
    g_per_Msun = 1.988*10**33
    cm_per_Rsun = 1.436 * 10**(-11)

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

        # MESA interpolations we will need
        global rho_fit
        rho_fit = interp(prof.radius_cm, prof.rho)
        global T_fit
        T_fit = interp(prof.radius_cm, prof.temperature)
        global grav_fit
        grav_fit = interp(prof.radius_cm, prof.grav)
        global x_mass_fraction_H_fit
        x_mass_fraction_H_fit = interp(prof.radius_cm, prof.x_mass_fraction_H)
        global M_star_cgs
        M_star_cgs = prof.star_mass * g_per_Msun
        global R_star_cgs
        R_star_cgs = prof.photosphere_r * cm_per_Rsun

        # calculate DM temp
        # masses to test in GeV
        m_chi_sample = [0.00001, 0.000015, 0.00002, 0.00003, 0.00005, 0.00007, 0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000, 15000]

        # masses to test in g
        m_chi_sample_cgs = []
        for i in range(len(m_chi_sample)):
            m_chi_sample_cgs.append(g_per_GeV * m_chi_sample[i])

        # use central temp to guess
        T_chi_guess = prof.temperature[-1]
        T_chi_sample = []

        # do MESA calcs and run thru all massses
        # for i in range(len(m_chi_sample)):
            # print("Solving Tchi for m_chi =", m_chi_sample[i], "GeV...")
            # use grams
            # m_chi = m_chi_sample_cgs[i]
            # R_star = R_star_cgs
            # T_chi_sample.append(fsolve(SP85_EQ410, T_chi_guess, args=(m_chi, R_star))[0])

        # read DM temp from csv
        T_chi_csv = []
        m_chi_csv = []
        with open('TM4.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                T_chi_csv.append(float(row[1]))
                m_chi_csv.append(float(row[0])*g_per_GeV)

        # now fit interpolation functions to T_chi w.r.t m_chi
        Tchi_fit = interp(m_chi_csv, T_chi_csv)

        r = np.linspace(prof.radius_cm[0], prof.radius_cm[-1], 100)
        n_chi_sample = []
        for i in range(len(r)):
            n_chi_sample.append(n_chi(r[i], T_chi_csv[12], m_chi_csv[12]))
        print(quad(n_chi, 0, prof.radius_cm[0], args=(T_chi_csv[12], m_chi_csv[12]))[0])

        plt.plot(r, n_chi_sample, label=mesa_lab)
        plt.title("MESA DM Density $100 M_{\\odot}$ (Windhorst)")
        plt.legend()
        plt.xlabel('$r$ [cm]')
        plt.ylabel('$n_{\\chi}$ [m$^{-3}$]')
        # plt.yscale("log")
        plt.xscale("log")
        # plt.show()
        plt.clf()

        # now calc evap rates
        evap_sample = []
        sigma = 1*10**(-43)
        for i in range(len(m_chi_csv)):
            print("Getting evap rate for m_chi =", m_chi_csv[i], "g...")
            evap_sample.append(evap_rate(T_chi_csv[i], m_chi_csv[i], sigma))

        m_chi_csv_GeV = []
        for i in range(len(m_chi_csv)):
            m_chi_csv_GeV.append(m_chi_csv[i]/g_per_GeV)

        # evap
        plt.plot(m_chi_csv_GeV, evap_sample, ls = '-', linewidth = 1, label=mesa_lab)
        plt.title("MESA DM Evap. Rate $100 M_{\\odot}$ (Windhorst)")
        plt.legend()
        plt.xlabel('$m_{\\chi}$ [Gev]')
        plt.ylabel('$E$ [$s^{-1}$]')
        plt.yscale("log")
        plt.xscale("log")
        plt.show()
        plt.clf()

###########
# EXECUTE #
###########
if __name__ == "__main__":
    # execute only if run as a script
    main()
