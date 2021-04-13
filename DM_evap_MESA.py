#!/usr/bin/env python
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
import matplotlib.colors as colors
import matplotlib
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import time
import csv
import copy
import os.path
from os import path

##################
# DEFINE CLASSES #
##################
class PopIIIStar:
    '''Describes important parameters of a population III star,
    Units:
            M - Solar
            R - Solar
            L - Solar
            Tc - Kelvin (K)
            rhoc - g/cm^3
            life_star - years'''

    def __init__(self, M = 0, R = 0, L = 0, Tc = 0, rhoc = 0, life_star = 0):
        self.mass = M
        self.radius = R
        self.lum = L
        self.core_temp = Tc
        self.core_density = rhoc
        self.lifetime = life_star

    # calculates stellar volume
    def get_vol(self):
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
def interp(x, y):
    '''takes two mesa data arrays and fits an interoplation function'''
    fit = interpolate.interp1d(x, y, fill_value="extrapolate")
    return fit

def R311(r, T_chi, m_chi, sigma):
    '''Eq. 3.11 from Goulde 1987, normalized evap. rate'''
    #TODO numerical integration over phase space
    a1 = (2/np.pi)*(2*k_cgs*T(r)/m_chi)**(1/2)
    a2 = (T(r)/T_chi)**(3/2)* sigma * n_p(r) * n_chi(r, T_chi, m_chi)
    b3 = np.exp(-1*(mu_plus(mu(m_chi))/xi(r, m_chi, T_chi))**2 *(m_chi*v_esc(r)**2/(2*T_chi)))
    c4 = mu(m_chi) * mu_minus(mu(m_chi)) / (T(r)*mu(m_chi)*xi(r, m_chi, T_chi)/T_chi)
    d5 = (xi(r, m_chi, T_chi)**2) / (T(r)*mu(m_chi)/T_chi)
    d6 = mu_plus(mu(m_chi)) * mu_minus(mu(m_chi)) / (mu(m_chi))
    c7 = (mu_plus(mu(m_chi))**3) / (xi(r, m_chi, T_chi) * ( (T(r)*mu(m_chi)/T_chi - mu(m_chi))))
    b8 = chi(gamma('-',  r, m_chi, T_chi), gamma('+',  r, m_chi, T_chi))
    b9 = np.exp(-1* m_chi * v_c(r)**2 / (2*T_chi) * (mu(m_chi)*T_chi) / (T(r)*mu(m_chi)))
    c10 = alpha('+', r, m_chi, v_c(r), v_esc(r)) * alpha('-', r, m_chi, v_c(r), v_esc(r))
    c11 = 1/(2*mu(m_chi))
    c12 = mu_minus(mu(m_chi))**2 *(1/mu(m_chi)) - (T_chi/(T(r)*mu(m_chi)))
    b13 = chi(alpha('-', r, m_chi, v_c(r), v_esc(r)), alpha('+', r, m_chi, v_c(r), v_esc(r)))
    b14 = np.exp(-1* m_chi * v_c(r)**2 / (2*T_chi)) * np.exp(-1*((m_chi*v_esc(r)**2 /2) - (m_chi*v_c(r)**2 /2))/T(r))
    b15 = mu_plus(mu(m_chi))**2 / ((T(r)*mu(m_chi)/T_chi) - mu(m_chi))
    b16 = chi(beta('-', r, m_chi, v_c(r), v_esc(r)), beta('+', r, m_chi, v_c(r), v_esc(r)))
    b17 = np.exp(-1 * (m_chi* v_chi(r, m_chi, T_chi)**2)/(2*T_chi) * alpha('-', r, m_chi, v_c(r), v_esc(r))**2)
    b18 = mu(m_chi) * alpha('+', r, m_chi, v_c(r), v_esc(r)) / (2*T(r)*mu(m_chi)/T_chi)
    b19 = np.exp(-1 * (m_chi* v_chi(r, m_chi, T_chi)**2)/(2*T_chi) * alpha('+', r, m_chi, v_c(r), v_esc(r))**2)
    b20 = mu(m_chi) * alpha('-', r, m_chi, v_c(r), v_esc(r)) / (2*T(r)*mu(m_chi)/T_chi)
    return a1*a2*(b3*(c4*(d5 - d6) + c7)*b8 + b9*(c10 - c11 + c12)*b13 - b14*b15*b16 - b17*b18 + b19*b20)

def R310(r, T_chi, m_chi, sigma):
    '''Eq. 3.10 from Goulde 1987, normalized evap. rate'''
    # compute each individual term
    a1 = (2/np.pi) * np.sqrt((2*k_cgs*T(r))/(m_chi))
    a2 = sigma * n_p(r) * n_chi(r, T_chi, m_chi)
    b3 = np.exp(-1*E_e(m_chi, v_esc(r))/ T(r))
    c4 = -1 * beta('+', r, m_chi, v_c(r), v_esc(r)) * beta('-', r, m_chi, v_c(r), v_esc(r))
    c5 = 1/(2*mu(m_chi))
    b6 = chi(beta('-', r, m_chi, v_c(r), v_esc(r)), beta('+', r, m_chi, v_c(r), v_esc(r)))
    b7 = np.exp(-1*E_c(m_chi, v_esc(r))/ T(r))
    c8 = alpha('+', r, m_chi, v_c(r), v_esc(r)) * alpha('-', r, m_chi, v_c(r), v_esc(r))
    c9 = 1/(2*mu(m_chi))
    b10 = chi(alpha('-', r, m_chi, v_c(r), v_esc(r)), alpha('+', r, m_chi, v_c(r), v_esc(r)))
    b11 = np.exp( D( (-1*E_c(m_chi, v_esc(r))/ T_chi) +  alpha('-', r, m_chi, v_c(r), v_esc(r))**2 ))
    ### TODO: overflow in B11???
    # print("b11 =",b11)
    # print("alpha- =", alpha('-', r, m_chi, v_c(r), v_esc(r)) )
    # print("term inside exp =" ,(-1*E_c(m_chi, v_esc(r))/ T_chi) +  alpha('-', r, m_chi, v_c(r), v_esc(r))**2)
    # print("left term:", -1*E_c(m_chi, v_esc(r))/ T_chi)
    # print("alpha^2", alpha('-', r, m_chi, v_c(r), v_esc(r))**2)
    b12 = np.sqrt((m_chi)/(2*k_cgs*T(r)))
    b13 = (v_esc(r) - v_c(r))/ 2
    # TODO overflow error in b14
    # print(D( (-1*E_c(m_chi, v_esc(r))/ T_chi) +  alpha('+', r, m_chi, v_c(r), v_esc(r))**2 ))
    # print("alpha+ =", alpha('+', r, m_chi, v_c(r), v_esc(r)) )
    b14 = np.exp( D( (-1*E_c(m_chi, v_esc(r))/ T_chi) +  alpha('+', r, m_chi, v_c(r), v_esc(r))**2 ))
    # print("b14 = ", b14)
    b15 = np.sqrt((m_chi)/(2*k_cgs*T(r)))
    b16 = (v_esc(r) + v_c(r))/ 2

    # skeleton structure of the expresion
    b11thru16 = b14*D(b15)*D(b16) - b11*D(b12)*D(b13)
    b11thru16 = float(b11thru16)
    return a1*a2*(b3*(c4 - c5)*b6 + b7*(c8-c9)*b10 + b11thru16)

def R311_2(r, T_chi, m_chi, sigma):
    '''Eq. 3.11 from Goulde 1987, normalized evap. rate'''
    #TODO numerical integration over phase space
    a1 = (2/np.pi) * np.sqrt((2*k_cgs*T(r))/(m_chi))
    a2 = (T(r)/T_chi)**(1.5) * sigma * n_p(r) * n_chi(r, T_chi, m_chi)
    b3 = np.exp(-1* (mu_plus(mu(m_chi)) / xi_2(r, m_chi, T_chi))**2 * (E_e(m_chi, v_esc(r))/T_chi))
    c4 = (mu(m_chi) * mu_minus(mu(m_chi))) / (nu(r, m_chi, T_chi) * xi_2(r, m_chi, T_chi))
    d5 = xi_2(r, m_chi, T_chi)**2 / nu(r, m_chi, T_chi)
    d6 = (mu_plus(mu(m_chi)) * mu_minus(mu(m_chi))) / (mu(m_chi))
    c7 = (mu_plus(mu(m_chi))**3 )/( xi_2(r, m_chi, T_chi) * (nu(r, m_chi, T_chi) - mu(m_chi)))
    b8 = chi(gamma_2('-', r, m_chi, T_chi, v_c(r), v_esc(r)), gamma_2('+', r, m_chi, T_chi, v_c(r), v_esc(r)))
    b9 = np.exp(-1*E_c(m_chi, v_c(r)) / T_chi) * (mu(m_chi) / nu(r, m_chi, T_chi))
    c10 = alpha('+', r, m_chi, v_c(r), v_esc(r)) * alpha('-', r, m_chi, v_c(r), v_esc(r))
    c11 = 1/(2*mu(m_chi))
    c12 = mu_minus(mu(m_chi))**2 * (1/(mu(m_chi)) - 1/(nu(r, m_chi, T_chi)))
    b13 = chi(alpha('-', r, m_chi, v_c(r), v_esc(r)), alpha('+', r, m_chi, v_c(r), v_esc(r)))
    b14 = np.exp(-1 * E_e(m_chi, v_esc(r)) /T_chi) * np.exp(-1 * (E_e(m_chi, v_esc(r)) - E_c(m_chi, v_c(r)))/T_chi)
    b15 = (mu_plus(mu(m_chi)))**2 /(nu(r, m_chi, T_chi) - mu(m_chi))
    b16 = chi(beta('-', r, m_chi, v_c(r), v_esc(r)), beta('+', r, m_chi, v_c(r), v_esc(r)))
    b17 = np.exp(-1 * (E_c(m_chi, v_c(r))/T_chi + alpha('-',  r, m_chi, v_c(r), v_esc(r))**2))
    b18 = mu(m_chi)/(2* nu(r, m_chi, T_chi)) * alpha('+',  r, m_chi, v_c(r), v_esc(r))
    b19 = np.exp(-1 * (E_c(m_chi, v_c(r))/T_chi + alpha('+',  r, m_chi, v_c(r), v_esc(r))**2))
    b20 = mu(m_chi)/(2* nu(r, m_chi, T_chi)) * alpha('-',  r, m_chi, v_c(r), v_esc(r))
    # print("a1 = ", a1)
    # print("a2 = ", a2)
    # print("b3 = ", b3)
    # print("c4 = ", c4)
    # print("d5 = ", d5)
    # print("d6 = ", d6)
    # print("c7 = ", c7)
    # print("b8 = ", b8)
    # print("b9 = ", b9)
    # print("c10 = ", c10)
    # print("c11 = ", c11)
    # print("c12 = ", c12)
    # print("b13 = ", b13)
    # print("b14 = ", b14)
    # print("b15 = ", b15)
    # print("b16 = ", b16)
    # print("b17 = ", b17)
    # print("b18 = ", b18)
    # print("b19 = ", b19)
    # print("b20 = ", b20)
    # print( a1*a2*(b3*(c4*(d5 - d6) + c7)*b8 + b9*(c10 - c11 + c12)*b13 - b14*b15*b16 - b17*b18 + b19*b20))
    return a1*a2*(b3*(c4*(d5 - d6) + c7)*b8 + b9*(c10 - c11 + c12)*b13 - b14*b15*b16 - b17*b18 + b19*b20)

def R311_3(mx, xi, sigma, star):
    #Central proton number density (cm^-3)
    nc = polytrope3_rhoc(star)*0.75/1.6726e-24
    kb = 1.380649e-16 #Boltzmann constant in cgs Units (erg/K)
    T = (10**8) * theta(xi)
    Tx = tau_fit(mx, star) * 10**8
    mu_p = mu_plus_minus('+', mx)
    mu_m = mu_plus_minus('-', mx)
    nu = nu_gould(mx, xi, star)
    xi_g = xi_gould(mx, xi, star)
    Ee = E_e_gould(mx, xi, star)
    Ec = E_c_gould(mx, xi, star)
    g_p = gamma_gould('+', mx, xi, star)
    g_m = gamma_gould('-', mx, xi, star)
    a_p = alpha('+', mx, vesc_r_poly(xi, star), vesc_r_poly(xi, star), xi, star)
    a_m = alpha('-', mx, vesc_r_poly(xi, star), vesc_r_poly(xi, star), xi, star)
    b_p = beta('+', mx, vesc_r_poly(xi, star), vesc_r_poly(xi, star), xi, star)
    b_m = beta('-', mx, vesc_r_poly(xi, star), vesc_r_poly(xi, star), xi, star)
    R_pre = 2/np.pi * proton_speed(xi, star) * star.get_vesc_surf() * 1/np.sqrt(mu(mx)) * (T/Tx)**(3/2) * sigma * nc * eta_proton(xi)
    term1 = np.exp(-(mu_p/xi_g)**2 * (Ee/(kb*Tx))) * (mu(mx)*mu_m/(nu*xi_g)*(xi_g**2/nu - mu_p*mu_m/mu(mx)) + mu_p**3/(xi_g*(nu-mu(mx))))*chi_func(g_m, g_p) 
    term2 = np.exp(-Ec/(kb*Tx)) * mu(mx)/nu * ( a_p*a_m - 1/(2*mu(mx)) + mu_m**2*(1/mu(mx) - 1/nu) ) * chi_func(a_m, a_p)
    term3 = np.exp(-Ec/(kb*Tx)) * np.exp(-(Ee-Ec)/(kb*T)) * mu_p**2/(nu - mu(mx)) * chi_func(b_m, b_p)
    term4 = np.exp(-(Ec/(kb*Tx) + a_m**2)) * mu(mx)*a_p/(2*nu)
    term5 = np.exp(-(Ec/(kb*Tx) + a_p**2)) * mu(mx)*a_m/(2*nu)
    R_val = R_pre * (term1 + term2 - term3 - term4 + term5) * normalize_factor(mx, xi, star)
    return R_val

def R_gould_approx(m_chi, r, sigma, Rstar):
    R_val = 2*np.pi**(-1/2)*sigma* n_p(r) * np.sqrt((2*k_cgs*T(r))/(m_chi)) * v_esc(Rstar) * np.exp(-1*(v_esc(r)/v_chi(r))**2 )
    return R_val

def Omegaplus37(r, w, T_chi, m_chi, sigma):
    '''Eq. 3.7 from Goulde 1987'''
    #TODO: boltzman constant
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
    '''integrand from Eq. 3.9'''
    return f_w38(r, w, T_chi, m_chi)*Omegaplus37(r, w, T_chi, m_chi, sigma)

def R39(r, T_chi, m_chi, sigma):
    '''Eq. 3.9 from Goulde 1987'''
    return quad(R39_integrand, 0, np.inf, args=(r, T_chi, m_chi, sigma), limit=1000)[0]

def v_chi(r, m_chi, T_chi):
    '''DM velocity assuming an isothermal dist'''
    return np.sqrt(2*k_cgs*T_chi/m_chi)

def n_chi(r, T_chi, m_chi):
    '''normalized isotropic DM distribution using user supplied potential and DM temp (from MESA)'''
    return np.exp(-1.0*m_chi*phi_quick(r)/(k_cgs*T_chi))

def alpha(pm, r, m_chi, w, v):
    '''made up goulde function'''
    if (pm == '+'):
        val = (m_p/(2 * k_cgs * T(r)))**(1/2) * (mu_plus(mu(m_chi)) * v + mu_minus(mu(m_chi)) * w)
    if (pm == '-'):
        val = (m_p/(2 * k_cgs * T(r)))**(1/2) * (mu_plus(mu(m_chi)) * v - mu_minus(mu(m_chi)) * w)
    return val

def gamma_2(pm, r, m_chi, T_chi, w, v):
    '''made up goulde function'''
    if (pm == '+'):
        val = np.sqrt(m_p / (2 * k_cgs * T(r))) * (grho(r, m_chi, T_chi)*v + xi_2(r, m_chi, T_chi)*w)
    if (pm == '-'):
        val = np.sqrt(m_p / (2 * k_cgs * T(r))) * (grho(r, m_chi, T_chi)*v - xi_2(r, m_chi, T_chi)*w)
    return val

def grho(r, m_chi, T_chi):
    '''made up goulde function'''
    val =  (mu_plus(mu(m_chi)) * (mu_minus(mu(m_chi))))/ xi_2(r, m_chi, T_chi)
    return val

def xi_2(r, m_chi, T_chi):
    '''not the polytope xi!!!!!!, just a made up goulde function'''
    ###TODO: invalid root???
    val =  np.sqrt(mu_minus(mu(m_chi))**2 + nu(r, m_chi, T_chi))
    return val

def nu(r, m_chi, T_chi):
    '''made up goulde function'''
    val =  mu(m_chi) * T(r) / T_chi
    return val

def E_e(m_chi, v):
    '''escape energy'''
    val =  m_chi * v**2 /2
    return val

def E_c(m_chi, w):
    '''cutoff energy'''
    val =  m_chi * w**2 /2
    return val

def beta(pm, r, m_chi, w, v):
    '''made up goulde function'''
    ###TODD: boltzmand constant???
    if (pm == '+'):
        val = (m_p/(2 * k_cgs * T(r)))**(1/2) * (mu_minus(mu(m_chi)) * v + mu_plus(mu(m_chi)) * w)
    if (pm == '-'):
        val = (m_p/(2 * k_cgs * T(r)))**(1/2) * (mu_minus(mu(m_chi)) * v - mu_plus(mu(m_chi)) * w)
    return val

def gamma(pm, r, m_chi, T_chi):
    '''made up goulde function'''
    ###TDOD: boltzmand constant???
    if (pm == '+'):
        val = (m_p/(2 * k_cgs * T(r)))**(1/2) * ((mu_plus(mu(m_chi)) * mu_minus(mu(m_chi)) )*v_esc(r)/xi(r, m_chi, T_chi) + xi(r, m_chi, T_chi)*v_c(r))
    if (pm == '-'):
        val = (m_p/(2 * k_cgs * T(r)))**(1/2) * ((mu_plus(mu(m_chi)) * mu_minus(mu(m_chi)) )*v_esc(r)/xi(r, m_chi, T_chi) - xi(r, m_chi, T_chi)*v_c(r))
    return val

def xi(r, m_chi, T_chi):
    '''not the polytope xi!!!!!!, just a made up goulde function'''
    ###TODO: wrong
    return np.sqrt(mu_minus(mu(m_chi))**2 + T(r)/(T_chi*mu(m_chi)))

def chi(a, b):
    '''made up goulde function'''
    return np.sqrt(np.pi)/2 * (sc.erf(b) - sc.erf(a))

def mu(m_chi):
    '''dimensional less DM mass'''
    return m_chi / m_p

def mu_plus(mu):
    '''dimensionless mass functions'''
    return (mu + 1)/ 2

def mu_minus(mu):
    '''dimensionless mass functions'''
    return (mu - 1 )/ 2

def v_c(r):
    '''cutoff velocity for the boltzman distribution'''
    return v_esc(r)

def v_esc(r):
    '''escape velocity at an arbitray point in the star'''
    #TDOD: prove this
    return np.sqrt(2*(G_cgs * M_star_cgs/R_star_cgs + (phi_quick(R_star_cgs) - phi_quick(r))))

# # FAKE POLY
# def v_esc(r):
#     '''Escape velocity of n=3 polytrope at given radius (dimensionless xi)'''
#     G = 6.6743*10**(-8) # gravitational constant in cgs units
#     xi = 6.89*(r / star.get_radius_cm())
#     r1 = star.get_radius_cm()
#     vesc = np.sqrt( 2*G*star.get_mass_grams()/star.get_radius_cm() + 2*(phi_poly(r1, star) - phi_poly(r, star)) )
#     return vesc

def T(r):
    '''temperature interpolation function that is fit to a MESA data array'''
    # TODO: swap with polytrope
    return T_fit(r)

# # FAKE POLY
# def T(r):
#     '''temperature from polytrope'''
#     xi = 6.89*(r / star.get_radius_cm())
#     return star.core_temp * theta(xi)

def n_p(r):
    '''number density interpolation function that uses fits from a MESA data array'''
    # TODO: swap with polytrope
    return x_mass_fraction_H_fit(r) * rho_fit(r) / m_p

# # FAKE POLY
# def n_p(r):
#     '''takes eta of  a N=3 polytrope, and then coverts to number density'''
#     n_p = rho_c_poly(star) * eta_poly(r, star) / m_p
#     return n_p

def phi_integrand(r):
    '''integrand for the phi() function'''
    #TODO: integrate over mass to calculate acceleration without using the acc parameter from MESA
    #TODO: integrate 0->r
    return 0.5* grav_fit(r)

def phi(r):
    ''' calculate potential from acceleration given by mesa'''
    # TODO: swap with polytrope
    return quad(phi_integrand, 0, r, limit=1000)[0]

def phi2_integrand(r):
    '''integrand for the phi2() function'''
    return G_cgs*mass_enc(r)/(r**2)

def phi2(r):
    ''' calculate potential from acceleration at radius r given by mesa'''
    return quad(phi2_integrand, 0, r, limit=1000)[0]

def phi_quick(r):
    '''uses an interpolation for the gravitational potential which is faster than taking the integrals everytime'''
    return phi_fit(r)

# # FAKE POLY
# def phi_quick(r):
#     '''polytropic potential'''
#     xi = 6.89*(r / star.get_radius_cm())
#     G = 6.6743*10**(-8) # gravitational constant in cgs units
#     phi_xi = 4*np.pi*G*(rho_c_poly(star))*(star.get_radius_cm()/xis[-1])**2 * (1 - theta(xi)) #cgs units
#     return phi_xi

def rho(r):
    '''calculates the density at r from MESA using an interpolated fit from a data array'''
    # TODO: swap with polytrope
    return rho_fit(r)

# # FAKE POLY
# def rho(r):
#     '''Density at radius r of polytrope'''
#     xi = 6.89*(r / star.get_radius_cm())
#     return rho_c_poly(star) * theta_cube(xi)

def mass_enc(r):
    '''calculates the mass enclosed to r from MESA using an interpolated fit from a data array'''
    return mass_enc_fit(r)
    # return g_per_Msun*mass_enc_fit(r)

def SP85_EQ410_integrand(r, T_chi, m_chi):
    '''the integrand that SP85_EQ410() will evaluate'''
    t1 = n_p(r)
    t2 = np.sqrt((m_p* T_chi + m_chi * T(r)) / (m_chi*m_p))
    t3 = T(r) - T_chi
    t4 = np.exp((-1 * m_chi * phi_quick(r))/(k_cgs * T_chi))
    return t1 * t2 * t3 * t4 * r**2

def SP85_EQ410(T_chi, m_chi, R_star):
    ''' uses MESA data in arrays, phi_mesa, n_mesa, prof.temperature, prof.radius_cm to evaluate the integral in EQ. 4.10 from SP85'''
    return quad(SP85_EQ410_integrand, 0, R_star, args=(T_chi, m_chi), limit=1000)[0]

def normfactor(r, m_chi, T_chi):
    '''normalization factor'''
    ###TODO:check
    t1 = sc.erf(v_c(r)/v_chi(r, m_chi, T_chi))
    t2 = (2/np.sqrt(np.pi))*(v_c(r)/v_chi(r, m_chi, T_chi))
    t3 = np.exp(-1*v_c(r)**2 /(v_chi(r, m_chi, T_chi)**2))
    return t1 - t2*t3

def evap_rate_integrand(r, T_chi, m_chi, sigma):
    '''the integrand that evap_rate() will evaluate'''
    r311 = r**2 * n_chi(r, T_chi, m_chi) * abs(R311_2(r, T_chi, m_chi, sigma)) / normfactor(r, m_chi, T_chi)
    return r311

def evap_rate_lower_integrand(r, T_chi, m_chi):
    '''the lower integrand that evap_rate() will evaluate'''
    n = r **2 * n_chi(r, T_chi, m_chi)
    return n

def evap_rate(T_chi, m_chi, sigma):
    '''evaporation rate of DM for the whole star'''
    return quad(evap_rate_integrand, 0, R_star_cgs, args=(T_chi, m_chi, sigma))[0]/quad(evap_rate_lower_integrand, 0, R_star_cgs, args=(T_chi, m_chi), limit=1000)[0]

def read_in_T_chi(name):
    '''reads T_chi vs M_chi data from CSV files'''
    # read DM temp from csv
    T_chi_csv = []
    m_chi_csv = []
    with open(name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            T_chi_csv.append(float(row[1]))
            m_chi_csv.append(float(row[0])*g_per_GeV)

    # now fit interpolation functions to T_chi w.r.t m_chi
    T_chi_fit = interp(m_chi_csv, T_chi_csv)
    return (m_chi_csv, T_chi_csv, T_chi_fit)

def read_in_evap(name):
    '''reads T_chi vs M_chi data from CSV files'''
    # read DM temp from csv
    m_chi_csv = []
    evap_csv = []
    with open(name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            evap_csv.append(float(row[1]))
            m_chi_csv.append(float(row[0]))

    # now fit interpolation functions to T_chi w.r.t m_chi
    return (m_chi_csv, evap_csv)

def solve_T_chi(m_chi_sample, name):
    '''solves T_chi in the SP85 equation 4.10 using fsolve'''
    m_chi_sample_cgs = []
    for i in range(len(m_chi_sample)):
        m_chi_sample_cgs.append(g_per_GeV * m_chi_sample[i])

    # use central temp to guess DM temp
    T_chi_guess = T(0)
    T_chi_sample = []

    # do MESA calcs and run thru all massses
    for i in range(len(m_chi_sample)):
        print("Solving Tchi for m_chi =", m_chi_sample[i], "GeV...")
        m_chi = m_chi_sample_cgs[i]
        R_star = R_star_cgs
        T_chi_sample.append(fsolve(SP85_EQ410, T_chi_guess, args=(m_chi, R_star))[0])

    # write to CSV
    m_chi_sample = np.asarray(m_chi_sample)
    T_chi_sample = np.asarray(T_chi_sample)
    output = np.column_stack((m_chi_sample.flatten(), T_chi_sample.flatten()))
    np.savetxt(name + '.csv',output,delimiter=',')

    return T_chi_sample

def assign_const():
    '''sets up physical constants as global parameters'''
    global m_p
    m_p = 1.6726231 * 10 ** (-24) # grams
    global g_per_GeV
    g_per_GeV = 1.783 *10 ** (-24)
    global G_cgs
    G_cgs = 6.6743*10**(-8) #cgs
    global k_cgs
    k_cgs = 1.3807 * 10 ** (-16) # cm2 g s-2 K-1
    global g_per_Msun
    g_per_Msun = 1.988*10**33
    global cm_per_Rsun
    cm_per_Rsun = 6.957*10**10 # cm

def mesa_args(direc, profile):
    '''read in what mesa data file to use'''
    arg1 = direc
    arg = arg1.split('_')
    hist= "history_" + arg[0] + ".data"
    direc = mr.MesaLogDir(log_path=arg1, history_file=hist)
    prof = direc.profile_data(int(profile))

    # read info about the MESA star
    lab_mass = str(round(prof.star_mass, 3))
    year = str(round(prof.star_age, 3))
    model = str(round(prof.model_number, 3))
    mesa_lab = year + " yr, " + lab_mass + " $M_{\\odot}$, " + model
    return (prof, mesa_lab, lab_mass)

def mesa_interp(prof):
    '''MESA interpolations we will need, sets them as globals'''
    global rho_fit
    rho_fit = interp(prof.radius_cm, prof.rho)
    global T_fit
    T_fit = interp(prof.radius_cm, prof.temperature)
    global grav_fit
    grav_fit = interp(prof.radius_cm, prof.grav)
    global x_mass_fraction_H_fit
    x_mass_fraction_H_fit = interp(prof.radius_cm, prof.x_mass_fraction_H)
    global mass_enc_fit
    mass_enc_fit = interp(prof.radius_cm, prof.mass_grams)
    global M_star_cgs
    M_star_cgs = prof.star_mass * g_per_Msun
    global R_star_cgs
    R_star_cgs = prof.photosphere_r * cm_per_Rsun

#def rho_c_poly(star):
#    '''Density at center of polytrope'''
#    ###TODO: not sure which central density to use
#    # getting stellar params
#    Mstar = star.get_mass_grams() #grams
#    Rstar = star.get_radius_cm()  #cm
#    # x-intercept of the theta function
#    xi_1 = xis[-1]
#    # slope of laneEmden at Theta = 0
#    deriv_xi1 = theta.derivatives(xis[-1])[1]
#    # central polytropic density as per n=3 polytropic model
#    rhoc_poly = (-1/(4*np.pi)) * ((xi_1/Rstar)**3) * (Mstar/(xi_1**2)) * (deriv_xi1)**-1 #g/cm^3
#    return rhoc_poly

def rho_c_poly(star):
    '''Density at center of polytrope'''
    ###TODO: not sure which central density to use
    rhoc_poly = star.core_density
    return rhoc_poly

def rho_poly(r, star):
    '''Density at radius r of polytrope'''
    xi = 6.89*(r / star.get_radius_cm())
    return rho_c_poly(star) * theta_cube(xi)

def phi_poly(r, star):
    '''polytropic potential'''
    xi = 6.89*(r / star.get_radius_cm())
    G = 6.6743*10**(-8) # gravitational constant in cgs units
    phi_xi = 4*np.pi*G*(rho_c_poly(star))*(star.get_radius_cm()/xis[-1])**2 * (1 - theta(xi)) #cgs units
    return phi_xi

def eta_poly(r, star):
    '''dimensionless number density of proton distribution in n = 3 polytrope'''
    xi = 6.89*(r / star.get_radius_cm())
    eta_xi = theta_cube(xi)
    return eta_xi

def n_p_poly(r, star):
    '''takes eta of  a N=3 polytrope, and then coverts to number density'''
    n_p = rho_c_poly(star) * eta_poly(r, star) / m_p
    return n_p

def n_chi_poly(mx, r, star, Tx): #Normalized
    '''isotropic DM distribution using potential from n=3 polytrope'''
    xi = 6.89*(r / star.get_radius_cm())
    kb = 1.380649e-16 #Boltzmann constant in cgs Units (erg/K)
    # mx in g
    mx_g = mx*1.783e-24
    # numerical DM number density profile for each DM mass (normalized)
    nx_xi_val = np.exp(-mx_g*phi_poly(r, star)/(kb*Tx))
    return nx_xi_val

def read_in_poly(name):
    '''retrieves solution to laneEmden n=3'''
    xis = []
    theta_arr = []
    with open(name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            xis.append(float(row[0]))
            theta_arr.append(float(row[1]))
    theta_cube = UnivariateSpline(xis, np.array(theta_arr)**3, k = 5, s = 0)
    # interpolating points for theta function
    theta = UnivariateSpline(xis, theta_arr, k = 5, s = 0)
    return (xis, theta, theta_cube)

def T_poly(r, star):
    '''temperature from polytrope'''
    xi = 6.89*(r / star.get_radius_cm())
    return star.core_temp * theta(xi)

def v_esc_poly(r, star):
    '''Escape velocity of n=3 polytrope at given radius (dimensionless xi)'''
    G = 6.6743*10**(-8) # gravitational constant in cgs units
    xi = 6.89*(r / star.get_radius_cm())
    r1 = star.get_radius_cm()
    vesc = np.sqrt( 2*G*star.get_mass_grams()/star.get_radius_cm() + 2*(phi_poly(r1, star) - phi_poly(r, star)) )
    return vesc

########
# MAIN #
########
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--direc", help="directory containing MESA profile and history files")
    parser.add_argument("-p", "--profile", help="index of the profile to use", type=int)
    parser.add_argument("-T", "--TchiMchi", help="name of csv file to store T_chi data in after solving with Eq. 4.10 from Spergel and Press 1985", action='store_true')
    parser.add_argument("-M", "--MESA", help="plot stellar parameters from MESA", action='store_true')
    parser.add_argument("-P", "--poly", help="plot stellar parameters for N=3 polytope", action='store_true')
    parser.add_argument("-E", "--Evap", help="plot DM evap rate by calculating from MESA data files", action='store_true')
    parser.add_argument("-e", "--evapcsv", help="plot DM evap rate using previously calculated csv", action='store_true')
    parser.add_argument("-R", "--G311", help="plot Gould 3.11 equation", action='store_true')
    parser.add_argument("-H", "--heatmap", help="plot heatmaps for alpha, beta, and gamma", action='store_true')
    args = parser.parse_args()

    # DM nucleon cross section
    sigma = 1*10**(-43)

    # assign various physical constants as global variables
    assign_const()

    # set up some ploting stuff
    fig = plt.figure(figsize = (12,8))
    plt.style.use('fast')
    palette = plt.get_cmap('magma')
    palette1 = plt.get_cmap('viridis')
    # palette.set_over('white')
    # palette.set_under('white')

    if args.poly:
        '''polytrope defintions and initalizations we will need if we want to compare'''
        global star
        star = PopIIIStar(100, 10**0.6147, 10**6.1470, 1.176e8, 32.3, 10**6)
        M100 = PopIIIStar(100, 10**0.6147, 10**6.1470, 1.176e8, 32.3, 10**6)
        M300 = PopIIIStar(300, 10**0.8697, 10**6.8172, 1.245e8, 18.8, 10**6)
        M1000 = PopIIIStar(1000, 10**1.1090, 10**7.3047, 1.307e8, 10.49, 10**6)

        # read in numerical solution to lane-ememda eqaution from CSV file
        global xis
        global theta
        global theta_cube
        (xis, theta, theta_cube) = read_in_poly('Lane_Emden.csv')

    if args.direc and args.profile:
        '''this is the main part of main'''
        # read in MESA data from files specified in command line arguments
        (prof, mesa_lab, lab_mass) = mesa_args(args.direc, args.profile)

        # create interpolation functions from MESA's arrays 
        mesa_interp(prof)

        # ASSIGN ARRAYS FOR PLOTTING AND SAMPLING 
        # DM mass in GeV
        m_chi_sample = np.logspace(-4, 3, 100)

        # DM mass in grams
        m_chi_sample_cgs = []
        for i in range(len(m_chi_sample)):
            m_chi_sample_cgs.append(g_per_GeV * m_chi_sample[i])

        # radius in cm
        r = np.linspace(prof.radius_cm[-1], prof.radius_cm[0], 100)

        # set up an interpolation for phi that's faster than the integration
        global phi_fit
        phi_range = []
        for i in range(len(r)):
            phi_range.append(phi(r[i]))
        phi_fit = interp(r, phi_range)

        # solve for temp with SP85
        if args.TchiMchi:
            '''use fsolve and SP85 to find the DM temperature'''
            name = "TM" + str(args.direc) +"_" + str(args.profile) + ".csv"
            T_chi_sample = solve_T_chi(m_chi_sample, name)

            # calc tau and mu
            tau = []
            mu = []
            for i in range(len(m_chi_sample)):
                tau.append(T_chi_sample[i]/T(0))
                mu.append(m_chi_sample_cgs[i]/m_p)

            # plot DM temp vs mass 
            plt.plot(m_chi_sample, T_chi_sample, ls = '-', linewidth = 2, color=palette1(4/10), label=mesa_lab)
            plt.title("DM Temperature in a MESA star")
            plt.legend()
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel('$m_{\chi}$ [GeV]')
            plt.ylabel('$T$ [K]')
            file = "TM_" + str(args.direc) + "_" + str(args.profile) + ".png"
            plt.savefig(file, dpi=400)
            plt.clf()

            # plot DM temp vs mass 
            plt.plot(mu, tau, ls = '-', linewidth = 2, color=palette1(4/10), label=mesa_lab)
            plt.title("DM Temperature in a MESA star")
            plt.legend()
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel('$\mu$')
            plt.ylabel('$\tau$')
            file = "taumu_" + str(args.direc) + "_" + str(args.profile) + ".png"
            plt.savefig(file, dpi=400)
            plt.clf()
        else:
            '''otherwise just read in previously calculated data from files'''
            file = "TM" + str(args.direc) +"_" + str(args.profile) + ".csv"
            if path.exists(file) == True:
                (m_chi_csv, T_chi_csv, T_chi_fit) = read_in_T_chi(file)
            else:
                print("The DM temperature data for", args.direc, args.profile, "has yet to be computed.")
                print("To do so, simply run:")
                print(" ")
                print("./DM_evap_MESA.py -D", args.direc, "-p", args.profile, "-T")
                print(" ")
                print("This will generate the necesary data and save it in", file)
                exit()

        if args.MESA:
            '''setllar params from MESA'''
            # mass and DM temp to use for ploting
            m_chi = 10**(-1)
            T_chi = T_chi_fit(m_chi*g_per_GeV)
            T_sample = []
            rho_sample = []
            phi_sample = []
            n_p_sample = []
            n_chi_sample = []
            v_esc_sample = []
            for i in range(len(r)):
                T_sample.append(T(r[i]))
                rho_sample.append(rho(r[i]))
                phi_sample.append(phi(r[i]))
                n_p_sample.append(n_p(r[i]))
                n_chi_sample.append(n_chi(r[i], T_chi, m_chi*g_per_GeV))
                v_esc_sample.append(v_esc(r[i]))

            if args.poly:
                r_poly = np.linspace(prof.radius_cm[-1], M100.get_radius_cm(), 100)
                T_poly_sample = []
                rho_poly_sample = []
                phi_poly_sample = []
                n_p_poly_sample = []
                n_chi_poly_sample = []
                v_esc_poly_sample = []
                for i in range(len(r_poly)):
                    T_poly_sample.append(T_poly(r_poly[i], M100))
                    rho_poly_sample.append(rho_poly(r_poly[i], M100))
                    phi_poly_sample.append(phi_poly(r_poly[i], M100))
                    n_p_poly_sample.append(n_p_poly(r_poly[i], M100))
                    n_chi_poly_sample.append(n_chi_poly(m_chi, r_poly[i], M100, T_chi))
                    v_esc_poly_sample.append(v_esc_poly(r_poly[i], M100))

            # PLOT density
            plt.plot(r, rho_sample, ls = '-', linewidth = 2, color=palette1(4/10), label=mesa_lab)
            plt.plot(r_poly, rho_poly_sample, label="N=3 Polytrope", color=palette1(7/10), linewidth=2, ls='--')
            plt.title("Density: MESA (Windhorst) vs. N=3, $100 M_{\\odot}$")
            plt.legend()
            plt.xlabel("$r$ [cm]")
            plt.ylabel("$\\rho$ [$g/cm^3$]")
            file = "rho_" + str(args.direc) + "_" + str(args.profile) + ".png"
            plt.savefig(file, dpi=400)
            plt.clf()

            # PLOT temp
            plt.plot(r, T_sample, ls = '-', linewidth = 2, color=palette1(4/10), label=mesa_lab)
            plt.plot(r_poly, T_poly_sample, label="N=3 Polytrope", color=palette1(7/10), linewidth=2, ls='--')
            plt.title("Temperature: MESA (Windhorst) vs. N=3, $100 M_{\\odot}$")
            plt.legend()
            plt.xlabel('$r$ [cm]')
            plt.ylabel('$T$ [K]')
            file = "T_" + str(args.direc) + "_" + str(args.profile) + ".png"
            plt.savefig(file, dpi=400)
            plt.clf()

            # PLOT n_p
            plt.plot(r, n_p_sample, ls = '-', linewidth = 2, color=palette1(4/10), label=mesa_lab)
            plt.plot(r_poly, n_p_poly_sample, label="N=3 Polytrope", color=palette1(7/10), linewidth=2, ls='--')
            plt.title("Proton Number Density: MESA (Windhorst) vs. N=3, $100 M_{\\odot}$")
            plt.legend()
            # plt.yscale("log")
            # plt.xscale("log")
            plt.xlabel("$r$ [cm]")
            plt.ylabel("$n_p$ [$cm^{-3}$]")
            file = "np_" + str(args.direc) + "_" + str(args.profile) + ".png"
            plt.savefig(file, dpi=400)
            plt.clf()

            # Plot n_chi
            plt.plot(r, n_chi_sample, ls = '-', linewidth = 2, color=palette1(4/10), label=mesa_lab)
            plt.plot(r_poly, n_chi_poly_sample, label="N=3 Polytrope", color=palette1(7/10), linewidth=2, ls='--')
            plt.title("DM Number Density: MESA (Windhorst) vs. N=3, $100 M_{\\odot}$")
            plt.legend()
            # plt.yscale("log")
            # plt.xscale("log")
            plt.xlabel("$r$ [cm]")
            plt.ylabel("$n_{\chi}$ [$cm^{-3}$]")
            file = "nchi_" + str(args.direc) + "_" + str(args.profile) + ".png"
            plt.savefig(file, dpi=400)
            plt.clf()

            # PLOT v_esc
            plt.plot(r, v_esc_sample, ls = '-', linewidth = 2, color=palette1(4/10), label=mesa_lab)
            plt.plot(r_poly, v_esc_poly_sample, label="N=3 Polytrope", color=palette1(7/10), linewidth=2, ls='--')
            plt.title("Escape Velocity: MESA (Windhorst) vs. N=3, $100 M_{\\odot}$")
            plt.legend()
            plt.xlabel("$r$ [cm]")
            plt.ylabel("$v_{esc}$ [cm/s]")
            file = "vesc_" + str(args.direc) + "_" + str(args.profile) + ".png"
            plt.savefig(file, dpi=400)
            plt.clf()

            # PLOT gravitation pot
            plt.plot(r, phi_sample, ls = '-', linewidth = 2, color=palette1(4/10), label=mesa_lab)
            plt.plot(r_poly, phi_poly_sample, label="N=3 Polytrope", color=palette1(7/10), linewidth=2, ls='--')
            plt.title("Gravitational Potential: MESA (Windhorst) vs. N=3, $100 M_{\\odot}$")
            plt.legend()
            plt.xlabel("$r$ [cm]")
            plt.ylabel("$\phi$ [ergs/g]")
            file = "phi_" + str(args.direc) + "_" + str(args.profile) + ".png"
            plt.savefig(file, dpi=400)
            plt.clf()

        if args.heatmap:
            alpha_sample = np.zeros([len(r), len(m_chi_sample)])
            beta_sample = np.zeros([len(r), len(m_chi_sample)])
            gamma_sample = np.zeros([len(r), len(m_chi_sample)])
            for i in range(len(r)):
                for j in range(len(m_chi_sample)):
                    # print("Calculating alpha, i = ", i, "/", len(r), ", j = ", j, "/", len(m_chi_sample_cgs))
                    alpha_sample[i,j] = alpha("+", r[i], m_chi_sample_cgs[j], 0.5*v_esc(r[i]), v_esc(r[i]))
                    beta_sample[i,j] = beta("+", r[i], m_chi_sample_cgs[j], 0.5*v_esc(r[i]), v_esc(r[i]))
                    gamma_sample[i,j] = gamma_2("+", r[i], m_chi_sample_cgs[j], T_chi_fit(m_chi_sample_cgs[j]), 0.5*v_esc(r[i]), v_esc(r[i]))

            ### DEBUGGING
            print(
                "alpha = ",
                alpha("+", 1*10**11, g_per_GeV*10**(3), 0.5*v_esc(1*10**11), v_esc(1*10**11)),
                "beta = ",
                beta("+", 1*10**11, g_per_GeV*10**(3), 0.5*v_esc(1*10**11), v_esc(1*10**11)),
                "gamma = ",
                gamma_2("+", 1*10**11, g_per_GeV*10**(3), T_chi_fit(g_per_GeV*10**(3)), 0.5*v_esc(1*10**11), v_esc(1*10**11))
            )

            # alpha
            plt.pcolormesh(m_chi_sample, r, alpha_sample, cmap=palette1, shading='auto', edgecolors='face', norm=colors.LogNorm(vmin=abs(alpha_sample.min()), vmax=abs(alpha_sample.max())))
            cbar = plt.colorbar()
            # cbar.set_label('$\\log_{10} (\\rho_{plat.}/$ GeV cm$^{-3})$', fontsize = 13)
            # cbar.set_ticks(list(np.linspace(9, 19, 11)))
            plt.title("Gould Alpha Function: 10^2 GeV, 10^-43 cm^-2, 100 Msun")
            # plt.legend()
            plt.ylabel('$r$ [cm]')
            # plt.ylim([0, 1.2*10**17])
            plt.xlabel("$m_{\chi}$ [GeV]")
            # plt.yscale("log")
            plt.xscale("log")
            file = "alpha_" + str(args.direc) + "_" + str(args.profile) + ".png"
            plt.savefig(file, dpi=400)
            plt.clf()

            # beta
            plt.pcolormesh(m_chi_sample, r, beta_sample, cmap=palette1, shading='auto', edgecolors='face', norm=colors.LogNorm(vmin=abs(beta_sample.min()), vmax=abs(beta_sample.max())))
            cbar = plt.colorbar()
            # cbar.set_label('$\\log_{10} (\\rho_{plat.}/$ GeV cm$^{-3})$', fontsize = 13)
            # cbar.set_ticks(list(np.linspace(9, 19, 11)))
            plt.title("Gould Beta Function: 10^2 GeV, 10^-43 cm^-2, 100 Msun")
            # plt.legend()
            plt.ylabel('$r$ [cm]')
            # plt.ylim([0, 1.2*10**17])
            plt.xlabel("$m_{\chi}$ [GeV]")
            # plt.yscale("log")
            plt.xscale("log")
            file = "beta_" + str(args.direc) + "_" + str(args.profile) + ".png"
            plt.savefig(file, dpi=400)
            plt.clf()

            # gamma
            plt.pcolormesh(m_chi_sample, r, gamma_sample, cmap=palette1, shading='auto', edgecolors='face', norm=colors.LogNorm(vmin=abs(gamma_sample.min()), vmax=abs(gamma_sample.max())))
            cbar = plt.colorbar()
            # cbar.set_label('$\\log_{10} (\\rho_{plat.}/$ GeV cm$^{-3})$', fontsize = 13)
            # cbar.set_ticks(list(np.linspace(9, 19, 11)))
            plt.title("Gould Gamma Function: 10^2 GeV, 10^-43 cm^-2, 100 Msun")
            # plt.legend()
            plt.ylabel('$r$ [cm]')
            # plt.ylim([0, 1.2*10**17])
            plt.xlabel("$m_{\chi}$ [GeV]")
            # plt.yscale("log")
            plt.xscale("log")
            file = "gamma_" + str(args.direc) + "_" + str(args.profile) + ".png"
            plt.savefig(file, dpi=400)
            plt.clf()

        if args.G311:
            '''now calculate evap rates'''
            # assign mass to be used for these plots
            m = 10**(-1)

            # DEBUG
            # R311_2(0.28*10**11, T_chi_fit(10**(-2)*g_per_GeV), 10**(-2)*g_per_GeV, sigma)
            # print(R310(0.93*10**11, T_chi_fit(m*g_per_GeV), m*g_per_GeV, sigma))

            # check to see if 3.11 recovers 3.10
            # rrr = 0.6*10**11  # cm
            # print("#######################")
            # print("R values at T(r) = T_chi")
            # print("R 3.10 =", R310(rrr, T_chi_fit(m*g_per_GeV), m*g_per_GeV, sigma))
            # print("R 3.11 =", R311_2(rrr, T_chi_fit(m*g_per_GeV), m*g_per_GeV, sigma))
            # print("#######################")

            R311_sample = []
            R310_sample = []
            norm = []
            tsame = []
            rate = []
            for i in range(len(r)):
                R310_sample.append(abs(R310(r[i], T_chi_fit(m*g_per_GeV), m*g_per_GeV, sigma)))
                R311_sample.append(abs(R311_2(r[i], T_chi_fit(m*g_per_GeV), m*g_per_GeV, sigma)))
                # R310_sample.append(R310(r[i], T_chi_fit(m*g_per_GeV), m*g_per_GeV, sigma))
                # R311_sample.append(R311_2(r[i], T_chi_fit(m*g_per_GeV), m*g_per_GeV, sigma))
                norm.append(normfactor(r[i], m*g_per_GeV, T_chi_fit(m*g_per_GeV)))
                rate.append(R311_sample[i]/norm[i])


            # print(R311_sample)
            # PLOT
            # plt.plot(r, rate, ls = '-', linewidth = 2, label=mesa_lab)
            plt.plot(r, rate, ls = '-', linewidth = 2, label="R 3.11 normalized " + mesa_lab)
            plt.plot(r, R311_sample, ls = '-', linewidth = 2, label="R 3.11 " + mesa_lab)
            plt.plot(r, R310_sample, ls = '--', linewidth = 2, label="R 3.10 " + mesa_lab)
            # plt.plot(r, rate, ls = '-', linewidth = 2, label=mesa_lab)
            # if tsame:
            #     plt.axvline(x=tsame[0], label="$T_{\chi} = T(r)$", c="#8A2BE2", linewidth=2)
            plt.title("MESA Gould Eq. 3.10 " + lab_mass  + "$M_{\\odot}$ (Windhorst)")
            plt.legend()
            plt.xlabel('$r$ [cm]')
            plt.ylabel('$R(w|v)$ [???]')
            plt.ylim(-10**(-9), 10**(-9))
            # plt.yscale("log")
            # plt.xscale("log")
            file = "R311_" + str(args.direc) + "_" + str(args.profile) + ".png"
            plt.savefig(file, dpi=400)
            # plt.show()
            plt.clf()

            plt.plot(r, norm, ls = '-', linewidth = 2, label=mesa_lab)
            plt.title("MESA Gould Normalization Factor" + lab_mass  + "$M_{\\odot}$ (Windhorst)")
            plt.legend()
            plt.xlabel('$r$ [cm]')
            plt.ylabel('')
            plt.yscale("log")
            # plt.xscale("log")
            file = "norm_" + str(args.direc) + "_" + str(args.profile) + ".png"
            plt.savefig(file, dpi=400)
            # plt.show()
            plt.clf()

        if args.Evap:
            '''NOW CALC EVAP RATES'''
            evap_sample = []
            for i in range(len(m_chi_csv)):
                print("####################################################")
                print("Getting evap. rate for m_chi =", m_chi_csv[i]/g_per_GeV, "GeV...")
                evap_sample.append(evap_rate(T_chi_csv[i], m_chi_csv[i], sigma))
                print("Evap. rate is =", evap_sample[i])

            m_chi_csv_GeV = []
            for i in range(len(m_chi_csv)):
                m_chi_csv_GeV.append(m_chi_csv[i]/g_per_GeV)

            print(evap_sample)

            # write to CSV
            m_chi_csv_GeV = np.asarray(m_chi_csv_GeV)
            evap_sample = np.asarray(evap_sample)
            output = np.column_stack((m_chi_csv_GeV.flatten(), evap_sample.flatten()))
            file = "E_" + str(args.direc) + "_" + str(args.profile) + ".csv"
            np.savetxt(file,output,delimiter=',')

            print(evap_sample)

            # PLOT
            plt.plot(m_chi_csv_GeV, evap_sample, ls = '-', linewidth = 1, label=mesa_lab)
            plt.title("MESA DM Evap. Rate $100 M_{\\odot}$ (Windhorst)")
            plt.legend()
            plt.xlim(10**-6, 10**3)
            plt.ylim(10**-15, 10**-4)
            plt.xlabel('$m_{\\chi}$ [Gev]')
            plt.ylabel('$E$ [$s^{-1}$]')
            plt.yscale("log")
            plt.xscale("log")
            file = "E_" + str(args.direc) + "_" + str(args.profile) + ".png"
            plt.savefig(file, dpi=400)
            # plt.show()
            plt.clf()

    if args.evapcsv:
        '''read and plt evap from csv'''
        file = "E_" + str(args.direc) +"_" + str(args.profile) + ".csv"
        if path.exists(file) == True:
            (m_chi_csv_GeV, evap_csv) = read_in_evap(file)
        else:
            print("The evaporation data for", args.direc, args.profile, "has yet to be computed.")
            print("To do so, simply run:")
            print(" ")
            print("./DM_evap_MESA.py -D", args.direc, "-p", args.profile, "-E")
            print(" ")
            print("This will generate the necesary data and save it in", file)
            exit()

        # PLOT
        plt.plot(m_chi_csv_GeV, evap_csv, ls = '-', linewidth = 1, label=mesa_lab)
        plt.title("MESA DM Evap. Rate $100 M_{\\odot}$ (Windhorst)")
        plt.legend()
        plt.xlim(10**-6, 10**3)
        plt.ylim(10**-15, 10**-4)
        plt.xlabel('$m_{\\chi}$ [Gev]')
        plt.ylabel('$E$ [$s^{-1}$]')
        plt.yscale("log")
        plt.xscale("log")
        file = "E_" + str(args.direc) + "_" + str(args.profile) + ".png"
        plt.savefig(file, dpi=400)
        # plt.show()
        plt.clf()


###########
# EXECUTE #
###########
if __name__ == "__main__":
    # execute only if run as a script
    main()
