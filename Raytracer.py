import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Geodesics import eq_of_motion, event_horizon, metric_tensor, R_s

G = 1
c = 1

def compute_U_t_photon(U_r, U_phi, r):
    """Set up the summation for the invariant quantity and then use that to find U_t for a photon."""
   
    g_00, g_11, g_33 = metric_tensor(r)

    U_t_squared = (-g_11 * U_r**2 - g_33 * U_phi**2) / g_00

    if U_t_squared < 0:
        raise ValueError("Initial velocities are too large.")

    U_t = np.sqrt(U_t_squared)

    return U_t

def initialise_raytracer(r_cam = 50 * R_s, impact_parameter = 0):
    """Generate a 2D image of the black hole shadow by ray tracing.
       The impact parameter is angular momentum over energy. Use geometrised units so energy = 1
       and angular momentum = impact parameter."""
    
    # A ray aimed directly at the centre will always be captured
    if abs(impact_parameter) < 1e-6:
        return 0.0, True

    # Calculate azimuthal component of velocity from angular momentum
    v_phi = impact_parameter / r_cam**2

    # Calculate radial component of velocity, ensuring it's negative since the photon is heading inward
    v_r = -np.sqrt(1.0 - (1.0 - R_s / r_cam) * impact_parameter**2 / r_cam**2)

    # Calculated time component of velocity at the camera position for a photon
    v_t = compute_U_t_photon(v_r, v_phi, r_cam)


    # Set up state vector and integrate
    A_0 = [0.0, r_cam, 0.0, v_t, v_r, v_phi]
    lambda_limits = (0, max_lambda)

    # Integrate the geodesics, but using lambda instead of tau
    sol = solve_ivp(eq_of_motion, lambda_limits, A_0, method='DOP853', events=event_horizon, max_step=1e12, rtol=1e-10, atol=1e-10)

    # Check if the photon was captured or escaped
    r_final = sol.y[1][-1]
    phi_final = sol.y[2][-1]
    captured = r_final <= 1.01 * R_s

    return phi_final, captured