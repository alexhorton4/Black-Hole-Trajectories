from Geodesics import initial_elliptical_conditions, integrator, initial_circular_conditions, initial_conditions, R_s
import numpy as np

G = 1
c = 1

def Kepler_validation(parameters):
    """Validate the trajectory of a circular orbit against Kepler's laws."""
    
    # Extract parameters
    M = parameters['M']
    radii = [n * R_s for n in range(5, 20, 1)]
    expected_periods = []
    measured_periods = []
    residuals = []
    
    for r_0 in radii:

        expected_T = 2 * np.pi * np.sqrt(r_0**3 / (G * M))
        tau_limits = (0, expected_T * 2.0)

        A_0 = initial_circular_conditions(r_0, t_0=0, phi_0=0)
        
        # Calculate actual orbital period from the trajectory
        trajectories = integrator(tau_limits, A_0)  # Integrate for the given radius

        phi = trajectories['phi']
        t_coord = trajectories['t']

        found = False
        for i in range(1, len(phi)):
            if (phi[i] - phi[0]) >= 2 * np.pi:  # Found one complete orbit
                measured_periods.append(t_coord[i] - t_coord[0])  # Coordinate time for one orbit
                expected_periods.append(2 * np.pi * np.sqrt(r_0**3 / (G * M)))  # Kepler's 3rd law
                found = True
                break

        # Calculate residuals as a percentage difference between measured and expected periods
        measured_periods_squared = np.array(measured_periods) ** 2
        expected_periods_squared = np.array(expected_periods) ** 2

        residuals = (measured_periods_squared - expected_periods_squared) / (expected_periods_squared) * 100

        if not found:
            print(f"Full orbit not completed for radius {r_0}.")
                
    return {'expected_periods': expected_periods, 'measured_periods': measured_periods, 'radii': radii, 'residuals': residuals}

def convergence_test(tau_limits, A_0):
    """Test convergence by running the same orbit at different tolerances
       and measuring the max error of the invariant."""
    
    tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
    max_invariant_error = []
    
    for tol in tolerances:
        traj = integrator(tau_limits, A_0, rtol=tol, atol=tol, n_eval=10000, max_step=1e12)
        
        invariant = traj['invariant']
        
        max_invariant_error.append(np.max(np.abs(invariant - invariant[0])))
    
    return {'tolerances': tolerances, 'invariant_error': max_invariant_error,}

