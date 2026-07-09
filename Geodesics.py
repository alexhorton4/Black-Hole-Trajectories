import numpy as np
from scipy.integrate import solve_ivp
from File_Reading import read_parameters, validation

all_orbits = read_parameters()
parameters = all_orbits[0]

G = 1
c = 1
R_s = 2*G*parameters['M']/c**2

def metric_tensor(r):
    """Set up the metric tensor components. g_11 is the time component,
       g_22 is the radial component and g_33 is the azimuthal component."""

    g_00 = (1 - (R_s / r))
    g_11 = -(1 / (1 - R_s / r))
    g_33 = -r**2

    return (g_00, g_11, g_33)

def specific_energy(r, v_t):
    """Function to compute the specific energy, which should be conserved."""

    g_00 = metric_tensor(r)[0]
    energy = g_00 * v_t
    return energy

def specific_angular_momentum(r, v_phi):
    """Function to compute the specific angular momentum  which should be conserved."""

    g_33 = metric_tensor(r)[2]
    ang_mom = - g_33 * v_phi
    return ang_mom

def compute_U_t(U_r, U_phi, r):
    """Set up the summation for the invariant quantity and then use that to find U_t.
      U is initial velocities and later on v is used for velocities at any given point."""
   
    g_00, g_11, g_33 = metric_tensor(r)

    U_t_squared = (c**2 - (g_11 * U_r**2) - (g_33 * U_phi**2)) / g_00

    if U_t_squared < 0:
        raise ValueError("Initial velocities are too large.")

    U_t = np.sqrt(U_t_squared)

    return U_t

def eq_of_motion(tau, A):
    """Set up the geodesic equations of motion.
       A is a list of 6 variables: the coordinates and velocities."""
    
    t, r, phi, v_t, v_r, v_phi = A

    dr_dtau = v_r
    dphi_dtau = v_phi
    dt_dtau = v_t

    dv_t_dtau = - (R_s / (r * (r - R_s))) * v_r * v_t
    dv_r_dtau = - ((R_s * c**2 * (r - R_s)) / (2 * r **3) * v_t**2) + ((R_s / (2 * r * (r - R_s))) * v_r**2) + ((r - R_s) * v_phi**2)
    dv_phi_dtau = - (2 / r) * v_r * v_phi

    return [dt_dtau, dr_dtau, dphi_dtau, dv_t_dtau, dv_r_dtau, dv_phi_dtau]

def event_horizon(tau, A):
    """Set up a function to check if the particle has crossed the event horizon. Use 1.01 to avoid
       massive numbers when r is close to R_s."""
    
    r = A[1]
    return r - (1.001 * R_s)

event_horizon.terminal = True
event_horizon.direction = -1 


def initial_conditions(r_0, t_0, phi_0, U_r_0, U_phi_0):
    """Initialise the starting values for all relevant variables, 
       the first t, r , phi, then U_t, U_r, U_phi"""

    if r_0 <= R_s:
        raise ValueError("Trajectory must start before the event horizon.")

    U_t_0 = compute_U_t(U_r_0, U_phi_0, r_0)
    A_0 = [t_0, r_0, phi_0, U_t_0, U_r_0, U_phi_0] # state vector

    return A_0

def initial_circular_conditions(r_0, t_0, phi_0):
    """Initialise the starting values for a circular orbit, 
       where dr/dtau = 0 and thus d^2r/dtau^2 = 0."""

    if r_0 <= 1.5 * R_s:
        raise ValueError("Circular orbits are only possible for before 1.5 Rs.")

    U_r_0 = 0
    U_phi_0 = (c / r_0) * np.sqrt((R_s)/ (2 * r_0 - 3 * R_s))
    U_t_0 = np.sqrt((2 * r_0 )/ (2 * r_0 - 3 * R_s)) 
    
    A_0 = [t_0, r_0, phi_0, U_t_0, U_r_0, U_phi_0] # state vector

    return A_0

def initial_elliptical_conditions(r_peri, eccentricity, phi_0, t_0):
    """Initialise the conditions for a particle with an elliptical orbit.
       Where r_peri is the radius at the periapsis."""
    
    if r_peri <= 1.5 * R_s:
        raise ValueError("Elliptical orbits are only possible for before 1.5 Rs.")
    if eccentricity < 0 or eccentricity >= 1:
        raise ValueError("Eccentricity must be between 0 and 1 for an elliptical orbit.")
    
    U_r_0 = 0
    U_phi_0 = (c / r_peri) * np.sqrt((R_s * (1 + eccentricity) / (2 * r_peri)))
    U_t_0 = np.sqrt((c**2 + r_peri**2 * U_phi_0**2) / (1 - R_s / r_peri))

    A_0 = [t_0, r_peri, phi_0, U_t_0, U_r_0, U_phi_0] # state vector

    return A_0

def integrator(tau_limits, A_0, method = 'DOP853', rtol = 1e-10, atol = 1e-10, max_step = 0.1, n_eval = 10000):

    tau_eval = np.linspace(tau_limits[0], tau_limits[1], n_eval)

    # Check that the final tau span is greater than the initial tau span
    if max_step <= 0:
        raise ValueError("max_step must be positive.")

    # Check that the boundary conditions won't result in a negative
    if tau_limits[1] <= tau_limits[0]:
        raise ValueError("Final tau must be greater than initial tau.")

    sol = solve_ivp(eq_of_motion, tau_limits, A_0, method = method, t_eval = tau_eval, events = event_horizon, max_step = max_step, rtol = rtol, atol = atol)
    
    tau = sol.t         # Proper time array
    t_coord = sol.y[0]  # Coordinate time
    r = sol.y[1]        # Radial coordinate
    phi = sol.y[2]      # Azimuthal angle
    v_t = sol.y[3]      # dt/dtau
    v_r = sol.y[4]      # dr/dtau
    v_phi = sol.y[5]    # dphi/dtau

    # Computes specific angular momentum at each timestep
    ang_mom = specific_angular_momentum(r, v_phi)

    # Computes specific energy at each timestep
    energy = specific_energy(r, v_t)

    # Computes invariant at each timestep
    g_00, g_11, g_33 = metric_tensor(r)
    invariant = g_00 * v_t**2 + g_11 * v_r**2 + g_33 * v_phi**2

    # Convert to Cartesian coordinates for plotting
    x_cartesian = r * np.cos(phi)
    y_cartesian = r * np.sin(phi)

    # Check if integration was successful and print messages accordingly
    if sol.success:
        print("Integration complete.")
    else:
        print("Integration failed: ", sol.message)

    # Stores all values in a dictionary
    trajectories = {
        'tau': tau,
        't': t_coord,
        'r': r,
        'phi': phi,
        'v_t': v_t,
        'v_r': v_r,
        'v_phi': v_phi,
        'angular_momentum': ang_mom,
        'energy': energy,
        'invariant': invariant,
        'x_cartesian': x_cartesian,
        'y_cartesian': y_cartesian,
        'Integration_method': method,
        'sol.success': sol.success,
        'R_s': R_s,
        'M': parameters['M'],
        'tau_length': len(tau)
    }
    return trajectories