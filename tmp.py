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


