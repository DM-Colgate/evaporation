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


##################
# DEFINE CLASSES #
##################
class PopIIIStar:
    '''
    describes important parameters of a population III star,
        units:
        M - Solar
        R - Solar
        L - Solar
        Tc - Kelvin (K)
        rhoc - g/cm^3
        life_star - years
    '''

    def __init__(self, M = 0, R = 0, L = 0, Tc = 0, rhoc = 0, life_star = 0):
        self.mass = M
        self.radius = R
        self.lum = L
        self.core_temp = Tc
        self.core_density = rhoc
        self.lifetime = life_star


    def get_vol(self):
        ''' calculates stellar volume '''
        vol = (4/3) * np.pi * (self.radius*6.96e10)**3 #in cm^3
        return vol


    def get_num_density(self):
        mn_grams = 1.6726e-24
        M = 1.9885e33 * self.mass
        n_baryon = 0.75*M/mn_grams * 1/(self.get_vol())
        return n_baryon


    def get_mass_grams(self):
        M_gram = 1.9885e33 * self.mass
        return M_gram

    def get_radius_cm(self):
        R_cm = self.radius*6.96e10
        return R_cm

    def get_vesc_surf(self):
        G  = 6.6743*10**(-8) #cgs units
        M = self.get_mass_grams()
        R = self.get_radius_cm()
        Vesc = np.sqrt(2*G*M/R) # escape velocity(cm/s) 
        return Vesc

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

def polytrope3_rhoc(star):
    '''density at center of polytrope'''
    # getting stellar params
    Mstar = star.get_mass_grams() #grams
    Rstar = star.get_radius_cm()  #cm

    # x-intercept of the theta function
    xi_1 = xis[-1]

    # slope of laneEmden at Theta = 0
    deriv_xi1 = theta.derivatives(xis[-1])[1]

    # central polytropic density as per n=3 polytropic model
    rhoc_poly = (-1/(4*np.pi)) * ((xi_1/Rstar)**3) * (Mstar/(xi_1**2)) * (deriv_xi1)**-1 #g/cm^3
    return rhoc_poly

def potential_poly(xi, star):
    '''Polytropic potential'''
    G = 6.6743*10**(-8) # gravitational constant in cgs units
    phi_xi = 4*np.pi*G*(polytrope3_rhoc(star)) * (star.get_radius_cm()/xis[-1])**2 * (1 - theta(xi)) #cgs units
    return phi_xi

def vesc_r_poly(xi, star):
    ''' escape velocity of n=3 polytrope at given radius (dimensionless xi) '''
    G = 6.6743*10**(-8) # gravitational constant in cgs units    
    xi1 =  xis[-1]
    vesc_xi = np.sqrt( 2*G*star.get_mass_grams()/star.get_radius_cm() + 2*(potential_poly(xi1, star) - potential_poly(xi, star)) )
    return vesc_xi/star.get_vesc_surf()

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

def mesa_proton_speed(Tchi, prof):
    u = []
    for i in range(len(Tchi)):
    u.append(np.sqrt(2*Tchi[i]/1.6726219e-24)) # cm/s (cgs units)
    return u

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

def mesa_alpha(plus_minus, mchi, Tchi, v, w, prof):
    u = proton_speed(Tchi, prof)
    alpha = []
    if (plus_minus == '+'):
        for i in range(len(Tchi)):
            alpha.append((mu_plus_minus('+', mchi)*v + mu_plus_minus('-', mchi)*w)/u)
    elif(plus_minus == '-'):
        for i in range(len(Tchi)):
            alpha.append((mu_plus_minus('+', mchi)*v - mu_plus_minus('-', mchi)*w)/u)
    return alpha

def mesa_beta(plus_minus, mchi, Tchi, v, w, prof):
    u = proton_speed(Tchi, prof)
    beta = []
    if (plus_minus == '+'):
        for i in range(len(Tchi)):
            beta.append((mu_plus_minus('-', mchi)*v + mu_plus_minus('+', mx)*w)/u)
    elif(plus_minus == '-'):
        for i in range(len(Tchi)):
            beta.append((mu_plus_minus('-', mchi)*v - mu_plus_minus('+', mchi)*w)/u)
    return beta

def chi_func(a,b):
    chi_val = np.sqrt(np.pi)/2 * (mp.erf(b) - mp.erf(a))
    return chi_val

def eta_proton(xi):
    ''' number density of proton distribution in n = 3 polytrope '''
    eta_xi = theta_cube(xi)
    return eta_xi

def R_plus(q, z, mx, xi, sigma, star):
    '''R_+ coefficient'''
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

def R_minus(q, z, mx, xi, sigma, star):
    '''R_- coefficient'''
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

def omega_plus(z, mx, xi, sigma, star):
    '''omega_plus function, integral over R_plus'''
    vesc = star.get_vesc_surf()
    omega_plus_val = vesc * mp.quad(lambda q: R_plus(q, z, mx, xi, sigma, star), [vesc_r_poly(xi, star), np.inf]) #s^-1
    return omega_plus_val

def omega(z, mx, xi, sigma, star):
    '''omega function, integral over R_plus + Rminus'''
    vesc = star.get_vesc_surf()
    omega_val = vesc * ( mp.quad(lambda q: R_minus(q, z, mx, xi, sigma, star), [vesc_r_poly(xi, star), z]) + mp.quad(lambda q: R_plus(q, z, mx, xi, sigma, star), [z, np.inf]) ) #s^-1
    return omega_val

def dm_speed(mx, star):
    '''average dm speed in star (isotropic)'''
    kb = 1.380649e-16 #Boltzmann constant in cgs Units (erg/K)
    Tx = (mx, star) * 10**8 #DM temperature
    mx_g = mx * 1.783e-24 #Converting GeV/c^2 to g
    vx = np.sqrt(2*kb*Tx/mx_g) #cm/s
    ux = vx/star.get_vesc_surf() #Dimensionless
    return ux

def mesa_dm_speed(mchi, Tchi, prof):
    '''average dm speed in star (isotropic)'''
    mchi_g = mchi * 1.783e-24 #Converting GeV/c^2 to g
    vchi = np.sqrt(2*kb*Tx/mx_g) #cm/s
    uchi = vchi/star.get_vesc_surf() #Dimensionless
    return uchi

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

def mesa_Vesc(phi):
    vesc = []
    factor = G*prof.star_mass*g_per_Msun/prof.radius_cm[0]
    for i in range(len(phi)):
        vesc.append(np.sqrt(2*factor*(phi[0] - phi[i])))
    return vesc

def mesa_R_integrand(v, w, mchi, Tchi, prof, i):
    # the bit in the [ ]'s in Ilie's jan 21 2021 notes 
    grossterm = (chi_func(mesa_alpha("-", mchi, Tchi, v, w, prof), mesa_alpha("+", mchi, Tchi, v, w, prof)) + chi_func(mesa_beta("-", mchi, Tchi, v, w, prof), mesa_beta("-", mchi, Tchi, v, w, prof))*np.exp(mu(mchi)*(w**2 - v**2)/(mesa_proton_speed(Tchi, prof))[i]**2))
    # the bit not in the [ ]'s
    lessgrossterm = 2/np.pi) * ((mu_plus_minus("+", mchi)**2)/mu(mchi)) * (v/w) * nchi[i] * sigma
    # take product
    return lessgrossterm*grossterm

def mesa_R_integrated(mchi, nchi, phi, prof, sigma, vcut)
    ''' returns a radial array, besides that idk whats going on here help'''
    Vesc = mesa_Vesc(phi)
    omega_p = []
    for i in range(len(phi)):
        # now we have to itegrate over velocity space..?!..?..
        omega_p.append(quad(lessgrossterm * grossterm, Vesc[i], numpy.inf, args=(w, mchi, Tchi, prof, i)))
    return omega_p

def nx_xi(mx, xi, star):
    ''' normalized isotropic DM distribution using potential from n=3 polytrope'''
    kb = 1.380649e-16 #Boltzmann constant in cgs Units (erg/K)
    # finding Tx using Temperature function
    Tx = tau_fit(mx, star) * 10**8 #K
    # mx in grams
    mx_g = mx*1.783e-24
    # numerical DM number density profile for each DM mass (normalized)
    nx_xi_val = np.exp(-mx_g*potential_poly(xi, star)/(kb*Tx))
    return nx_xi_val

def mesa_nchi(mchi, prof, phi, Tchi):
    '''
    normalized isotropic DM distribution using user supplied potential and DM temp (from MESA)
    phi and Tchi are the arrays generated from MESA for each of it's radial cells
    mchi is DM mass
    nchi returned is an array of DM number denisty in each mesa cell
    '''
    # mchi in grams
    mchi_g = mx*1.783e-24
    n_chi = []
    for all i in phi:
        # numerical DM number density profile for each DM mass (normalized)
        nchi.append(np.exp(-mchi_g*phi[i]/(k_cgs*Tchi)))
    return nchi

def mesa_fchi_integrated(mchi, prof, v, w):
    ux = dm_speed(mx, prof)
    ue_xi = vesc_r_poly(xi, star)
    vesc = star.get_vesc_surf()
    f_x_val = np.exp(-z**2/ux**2) * ( np.pi**(3/2) * ux**3 * vesc**3 * (sc.erf(ue_xi/ux) - 2/np.sqrt(np.pi)*ue_xi/ux*np.exp(-ue_xi**2/ux**2) )  )**-1
    return f_x_val


def upper_integrand(xi, mx, sigma, star, vcut_inf = False):
    return xi**2 * nx_xi(mx, xi, star) * R_integrated(mx, xi, sigma, star, vcut_inf)

def lower_integrand(xi, mx, sigma, star):
    return xi**2 * nx_xi(mx, xi, star)

def mesa_upper_integrand(mchi, sigma, phi, Tchi, prof, vcut):
    '''
    returns an array to be integrated wrt radius (?) vol
    '''
    # generate an array of DM number density
    nchi = nchi(mchi, prof, phi, Tchi)
    up = []
    for i in range(len(nchi)):
        up.append(nchi[i] * mesa_fchi_integrated(v, w)[i] * mesa_R_integrated(mchi, phi, prof, sigma, vcut)[i])
        # TODO radial array, volume integral.... hmmmm....
        # up.append(prof.radius_cm[i]**2 * nchi[i] * R_integrated(mchi, prof, sigma, vcut))
    return up

def mesa_lower_integrand(mchi, sigma, phi, Tchi, prof):
    '''
    returns an array to be integrated wrt radius (?)
    '''
    # generate an array of DM number density
    nchi = nchi(mchi, prof, phi, Tchi)
    low = []
    for i in range(len(nchi)):
        low.append(nchi[i])
        # TODO radial array, volume integral.... hmmmm....
        # low.append(prof.radius_cm**2 * nchi[i])
    return low

def evap_coeff(mx, sigma, star, vcut_inf = False):
    xi1 = xis[-1]
    E = quad(upper_integrand, 0, xi1, args=(mx, sigma, star, vcut_inf))[0]/quad(lower_integrand, 0, xi1, args=(mx, sigma, star))[0]
    return E

def mesa_evap_coeff(mx, sigma, prof, vcut):
    # number of cells in mesa
    nz = len(prof.radius)
    # take integrals radially across the entire star
    E = np.trapz(mesa_upper_integrand(mchi, sigma, phi, Tchi, prof, vcut), prof.radius_cm)/np.trapz(mesa_lower_integrand(mchi, sigma, phi, Tchi, prof), prof.radius_cm)
    return E

def retrieve_tau(star_mass):
    '''Retrieves tau(mx) from stored data'''
    mx = []
    tau = []
    with open('tau_mx_M%i.csv'%star_mass) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            mx.append(float(row[0]))
            tau.append(float(row[1]))
    return (mx, tau)

def retrieve_LaneEmden():
    '''Retrieves solution to laneEmden n=3'''
    xis = []
    theta_arr = []
    with open('Lane_Emden.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            xis.append(float(row[0]))
            theta_arr.append(float(row[1]))
    return (xis, theta_arr)

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

    # define some pop III polytopic stars
    poly100 = PopIIIStar(100, 10**0.6147, 10**6.1470, 1.176e8, 32.3, 10**6)
    poly300 = PopIIIStar(300, 10**0.8697, 10**6.8172, 1.245e8, 18.8, 10**6)
    poly1000 = PopIIIStar(1000, 10**1.1090, 10**7.3047, 1.307e8, 10.49, 10**6)
    stars_list = (poly100, poly300, poly1000)

    tau_fit_funcs = []
    mx_tau_fit, tau_temp = retrieve_tau(100)
    tau_fit_funcs.append(UnivariateSpline(mx_tau_fit, tau_temp, k = 5, s = 0))

    # polytrope conversion
    xis, theta_arr = retrieve_LaneEmden()
    theta = UnivariateSpline(xis, theta_arr, k = 5, s = 0)
    phi_xi_poly = potential_poly(xis, poly100)
    xis_frac = np.true_divide(xis, xis[-1])

    #Taking typical sigma value
    sigma = 1e-43

    #DM Mass
    mx = np.logspace(-4, 0, 30)

    # numerical vs Approximate solution
    E_G = []
    E_ilie = []
    E_ilie2 = []
    E_G.append([])
    E_ilie.append([])
    E_ilie2.append([])

    # loop over DM masses for poly and approx
    for i in range(0, len(mx)):
        E_G.append(evap_coeff_G(mx[i], sigma, star))
        E_ilie.append(evap_coeff_Ilie_approx(mx[i], sigma, star))
        E_ilie2.append(evap_coeff_Ilie_approx2(mx[i], sigma, star))

    # plot formatting
    fig = plt.figure(figsize = (12,8))
    plt.style.use('fast')
    palette = plt.get_cmap('viridis')

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

