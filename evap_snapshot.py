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
        r = prof.radius[k:]
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


def calc_Tchi(func, Tchi_guess):
    ''' returns for what Tchi the input function is zero'''
    return fsolve(func, Tchi_guess)


# misc evap functions (for polytropes)
def vesc_r_poly(xi, star):
    ''' escape velocity of n=3 polytrope at given radius (dimensionless xi) '''
    G = 6.6743*10**(-8) # gravitational constant in cgs units    
    xi1 =  xis[-1]
    vesc_xi = np.sqrt( 2*G*star.get_mass_grams()/star.get_radius_cm() + 2*(potential_poly(xi1, star) - potential_poly(xi, star)) )
    return vesc_xi/star.get_vesc_surf()


# mu functions
def mu(mx):
    ''' takes m_chi in GeV '''
    mu_val = mx/0.93827
    return mu_val


def mu_plus_minus(plus_minus, mx):
    if(plus_minus == '+'):
        mu_plus_val = (mu(mx) + 1)/2
    elif(plus_minus == '-'):
        mu_plus_val = (mu(mx) - 1)/2
    return mu_plus_val

def proton_speed(xi, star):
    ''' l(r), most probable dimensionless velocity of protons at specific point in star '''
    kb = 1.380649e-16 # boltzmann constant in cgs Units (erg/K)
    Tc = 10**8 # central star temperature taken to be ~ 10^8 K
    u = np.sqrt(2*kb*Tc*theta(xi)/1.6726219e-24) # cm/s (cgs units)
    l = u/star.get_vesc_surf()
    return l


# alpha/beta functions
def alpha(plus_minus, mx, q, z, xi, star):
    l = proton_speed(xi, star)
    if (plus_minus == '+'):
        alpha_val = (mu_plus_minus('+', mx)*q + mu_plus_minus('-', mx)*z)/l
    elif(plus_minus == '-'):
        alpha_val = (mu_plus_minus('+', mx)*q - mu_plus_minus('-', mx)*z)/l
    return alpha_val

def beta(plus_minus, mx, q, z, xi, star):
    l = proton_speed(xi, star)
    if (plus_minus == '+'):
        beta_val = (mu_plus_minus('-', mx)*q + mu_plus_minus('+', mx)*z)/l
    elif(plus_minus == '-'):
        beta_val = (mu_plus_minus('-', mx)*q - mu_plus_minus('+', mx)*z)/l
    return beta_val


## chi function ##
def chi_func(a,b):
    chi_val = np.sqrt(np.pi)/2 * (mp.erf(b) - mp.erf(a))
    return chi_val


def eta_proton(xi):
    ''' number density of proton distribution in n = 3 polytrope '''
    eta_xi = theta_cube(xi)
    return eta_xi


# R_+ coefficient
def R_plus(q, z, mx, xi, sigma, star):
    # dimensionless quantities
    mu_val = mu(mx)
    mu_plus = mu_plus_minus('+', mx)
    eta = eta_proton(xi)
    alpha_minus = alpha('-', mx, q, z, xi, star)
    alpha_plus = alpha('+', mx, q, z, xi, star)
    beta_minus = beta('-', mx, q, z, xi, star)
    beta_plus = beta('+', mx, q, z, xi, star)
    chi_alpha = chi_func(alpha_minus, alpha_plus)
    chi_beta = chi_func(beta_minus, beta_plus)
    l = proton_speed(xi, star)

    # central proton number density (cm^-3)
    nc = polytrope3_rhoc(star)*0.75/1.6726e-24

    # R_plus calculation
    R_plus_val = 2*nc/np.sqrt(np.pi) * mu_plus**2/mu_val * q/z * eta * sigma * (chi_alpha + chi_beta*mp.exp(mu_val * (z**2 - q**2)/(l**2))) # cm^-1

    return R_plus_val


# R_- coefficient
def R_minus(q, z, mx, xi, sigma, star):
    # dimensionless quantities
    mu_val = mu(mx)
    mu_plus = mu_plus_minus('+', mx)
    eta = eta_proton(xi)
    alpha_minus = alpha('-', mx, q, z, xi, star)
    alpha_plus = alpha('+', mx, q, z, xi, star)
    beta_minus = beta('-', mx, q, z, xi, star)
    beta_plus = beta('+', mx, q, z, xi, star)
    chi_alpha = chi_func(-1*alpha_minus, alpha_plus)
    chi_beta = chi_func(-1*beta_minus, beta_plus)
    l = proton_speed(xi, star)

    # central proton number density (cm^-3)
    nc = polytrope3_rhoc(star)*0.75/1.6726e-24

    #R_plus calculation
    R_minus_val = 2*nc/np.sqrt(np.pi) * mu_plus**2/mu_val * q/z * eta * sigma * (chi_alpha + chi_beta*mp.exp(mu_val * (z**2 - q**2)/(l**2))) # cm^-1

    return R_minus_val


# omega_plus function, integral over R_plus
def omega_plus(z, mx, xi, sigma, star):
    vesc = star.get_vesc_surf()
    omega_plus_val = vesc * mp.quad(lambda q: R_plus(q, z, mx, xi, sigma, star), [vesc_r_poly(xi, star), np.inf]) #s^-1
    return omega_plus_val


# omega function, integral over R_plus + Rminus
def omega(z, mx, xi, sigma, star):
    vesc = star.get_vesc_surf()
    omega_val = vesc * ( mp.quad(lambda q: R_minus(q, z, mx, xi, sigma, star), [vesc_r_poly(xi, star), z]) + mp.quad(lambda q: R_plus(q, z, mx, xi, sigma, star), [z, np.inf]) ) #s^-1
    return omega_val


# average dm speed in star (isotropic)
def dm_speed(mx, star):
    kb = 1.380649e-16 #Boltzmann constant in cgs Units (erg/K)
    Tx = (mx, star) * 10**8 #DM temperature
    mx_g = mx * 1.783e-24 #Converting GeV/c^2 to g
    vx = np.sqrt(2*kb*Tx/mx_g) #cm/s
    ux = vx/star.get_vesc_surf() #Dimensionless
    return ux


def f_x(z, mx, xi, star):
    # dm dimensionless speed
    ux = dm_speed(mx, star)
    ue_xi = vesc_r_poly(xi, star)
    vesc = star.get_vesc_surf()
    f_x_val = np.exp(-z**2/ux**2) * ( np.pi**(3/2) * ux**3 * vesc**3 * (sc.erf(ue_xi/ux) - 2/np.sqrt(np.pi)*ue_xi/ux*np.exp(-ue_xi**2/ux**2) )  )**-1
    return f_x_val


def f_x_inf(z, mx, xi, star):
    # dm dimensionless speed
    ux = dm_speed(mx, star)
    vesc = star.get_vesc_surf()
    f_x_val = mp.exp(-z**2/ux**2) * ( np.pi**(3/2) * ux**3 * vesc**3 )**-1
    return f_x_val


def integrand(z, mx, xi, sigma, star, vcut_inf = False):
    return f_x(z, mx, xi, star)*omega_plus(z, mx, xi, sigma, star)*z**2


def integrand_inf(z, mx, xi, sigma, star, vcut_inf = False):
    return f_x_inf(z, mx, xi, star)*omega(z, mx, xi, sigma, star)*z**2


def R_integrated(mx, xi, sigma, star, vcut_inf = False):
    vesc = star.get_vesc_surf()
    if(vcut_inf == False):
        R = 4*np.pi * vesc**3 * quad(integrand, 0, vesc_r_poly(xi, star), args=(mx, xi, sigma, star, vcut_inf))[0]
    else:
        R = 4*np.pi * vesc**3 * mp.quad(lambda z: integrand_inf(z, mx, xi, sigma, star, vcut_inf), [0, np.inf])
    return R


def upper_integrand(xi, mx, sigma, star, vcut_inf = False):
    return xi**2 * nx_xi(mx, xi, star) * R_integrated(mx, xi, sigma, star, vcut_inf)


def lower_integrand(xi, mx, sigma, star):
    return xi**2 * nx_xi(mx, xi, star)


def evap_coeff(mx, sigma, star, vcut_inf = False):
    xi1 = xis[-1]
    E = quad(upper_integrand, 0, xi1, args=(mx, sigma, star, vcut_inf))[0]/quad(lower_integrand, 0, xi1, args=(mx, sigma, star))[0]
    return E


#Retrieves tau(mx) from stored data
def retrieve_tau(star_mass):
    mx = []
    tau = []
    with open('tau_mx_M%i.csv'%star_mass) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            mx.append(float(row[0]))
            tau.append(float(row[1]))
    return (mx, tau)

def tau_fit(mx, star_mass): #Returns tau from fitting function based on star and dm mass
    if(mx > 100):
        tau_val = 1
    else:
        if(star_mass == 100):
            tau_val = tau_fit_funcs[0](mx)
        elif(star_mass == 300):
            tau_val = tau_fit_funcs[1](mx)
        elif(star_mass == 1000):
            tau_val = tau_fit_funcs[2](mx)
        else:
            tau_val = 1
    return tau_val

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
    parser.add_argument("-n", "--np", help="plot proton number denisty from MESA data files", action='store_true')

    args = parser.parse_args()

    # set variables
    global mchi
    mchi = 1.6726231 * 10 ** (-24) # grams
    global m_p
    m_p = 1.6726231 * 10 ** (-24) # grams
    global k_cgs
    k_cgs = 1.3807 * 10 ** (-16) # cm2 g s-2 K-1 
    global g_per_GeV
    g_per_GeV = 5.61 *10 ** (-23)

    # read in MESA data from files specified in command line arguments
    # arg1 = str(sys.argv[1])
    arg1 = args.direc
    arg = arg1.split('_')
    hist= "history_" + arg[0] + ".data"
    direc = mr.MesaLogDir(log_path=arg1, history_file=hist)
    # prof = direc.profile_data(int(sys.argv[2]))
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
    T_mesa = prof.temperature
    r_mesa = prof.radius

    # use central temp to guess
    Tchi_guess = T_mesa[-1]

    # masses to test
    mchi_sample = [0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 200, 300, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000]
    Tchi_sample = []

    # run thru all massses
    for i in range(len(mchi_sample)):
        mchi = g_per_GeV * mchi_sample[i]
        print("Solving Tchi for Mchi = ", mchi_sample[i], "...")
        # numerically solve
        Tchi_sample.append(calc_Tchi(SP85_EQ410, Tchi_guess))

    # convert to dimensionless
    mu_sample = []
    Tau_sample = []
    for i in range(len(mchi_sample)):
        mu_sample.append(g_per_GeV * mchi_sample[i] / m_p)
        Tau_sample.append(Tchi_sample[i] / T_mesa[-1])

    tau_fit_funcs = []
    mx_tau_fit, tau_temp = retrieve_tau(100)
    tau_fit_funcs.append(UnivariateSpline(mx_tau_fit, tau_temp, k = 5, s = 0))


    # plot
    if args.TchiMchi == True:
        plt.plot(mchi_sample, Tchi_sample, ls = '-', linewidth = 1, label="$100 M_{\odot}$")
        plt.title("MESA DM Temperature $100 M_{\odot}$ (Windhorst)")
        plt.legend()
        plt.xlabel('$M_{\chi}$ [Gev]')
        plt.ylabel('$T_{\chi}$ [K]')
        plt.yscale("log")
        plt.xscale("log")
        plt.show()
        plt.clf()

    # plot
    if args.taumu == True:
        plt.plot(mu_sample, Tau_sample, ls = '-', linewidth = 1, label="$100 M_{\odot}$")
        # plt.plot(mx_tau_fit, tau_temp, ls = '-', linewidth = 1, label="Poly $100 M_{\odot}$")
        plt.title("MESA DM Temperature $100 M_{\odot}$ (Windhorst)")
        plt.legend()
        plt.xlabel('$ \mu $')
        plt.ylabel('$ \tau $')
        plt.yscale("log")
        plt.xscale("log")
        plt.show()
        plt.clf()

    if args.phi == True:
        plt.plot(phi_mesa, r_mesa, ls = '-', linewidth = 1, label="$100 M_{\odot}$")
        plt.title("MESA Grav. Pot. $100 M_{\odot}$ (Windhorst)")
        plt.legend()
        plt.xlabel('$ \mu $')
        plt.ylabel('$ \tau $')
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

