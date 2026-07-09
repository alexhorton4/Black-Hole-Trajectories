from File_Reading import read_parameters, validation, write_results, select_orbit
from Geodesics import initial_circular_conditions, initial_elliptical_conditions, initial_conditions, integrator, R_s
from Plotting import (plot_trajectories, plot_kepler_validation, plot_energy_conservation,
                       plot_ang_mom_conservation, plot_invariant_conservation, plot_convergence, plot_time_dilation)
from Investigations import Kepler_validation, convergence_test

if __name__ == "__main__":

    all_orbits = read_parameters("parameters.txt")

    print("Please select an orbit from the following list or type 'no' to exit:")
    
    while True:
        params = select_orbit(all_orbits)
        if params is None:
            print("No orbit selected. Exiting program.")
            break
        validation(params)
        
        r_0 = params['r_0'] * R_s

        if params['orbit_type'] == 'circular':
            print("\nUsing circular initial conditions.")
            A_0 = initial_circular_conditions(r_0, t_0=0, phi_0=0)
        elif params['orbit_type'] == 'elliptical':
            print("\nUsing elliptical initial conditions.")
            r_peri = params['r_peri'] * R_s
            eccentricity = params['eccentricity']
            A_0 = initial_elliptical_conditions(r_peri, eccentricity, phi_0=0, t_0=0)
        elif params['orbit_type'] == 'custom':
            print("\nUsing custom initial conditions.")
            U_r_0 = params['v_r']
            U_phi_0 = params['v_phi']
            A_0 = initial_conditions(r_0, t_0=0, phi_0=0, U_r_0=U_r_0, U_phi_0=U_phi_0)
        
        # Run integration
        print("\nIntegrating geodesic equations...")
        tau_limits = (0, params['max_tau'])
        trajectories = integrator(tau_limits, A_0, method=str(params['method']), n_eval=int(params['n_eval']))
        
        orbit_name = params.get('name', 'Unknown')
        
        write_results(trajectories, orbit_name)
        print("Integration complete and results written to 'results.csv'.")

        # Plot trajectory
        print("\nGenerating trajectory plot...")
        plot_trajectories(trajectories, name = f"{params['name']}")
        print("Trajectory plot complete.")

        # Time dilation plot
        while True:
            td_input = input("\nDo you want to see the time dilation plot for this orbit? (yes/no): ").strip().lower()
            if td_input in ['yes', 'y']:
                plot_time_dilation(trajectories, orbit_name)
                break
            elif td_input in ['no', 'n']:
                print("Skipping time dilation plot.")
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")


        # Conservation plots
        while True:
            conservation_input = input("\nDo you want to see the conservation plots for the selected orbit? (yes/no): ").strip().lower()
            if conservation_input in ['yes', 'y']:
                plot_energy_conservation(trajectories, orbit_name)
                plot_ang_mom_conservation(trajectories, orbit_name)
                plot_invariant_conservation(trajectories, orbit_name)
                break
            elif conservation_input in ['no', 'n']:
                print("Skipping conservation plots.")
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

        # Convergence test
        while True:
            convergence_input = input("\nDo you want to run the convergence test for the selected orbit? (yes/no): ").strip().lower()
            if convergence_input in ['yes', 'y']:
                print("Running convergence test...")
                plot_convergence(convergence_test(tau_limits, A_0), orbit_name)
                break
            elif convergence_input in ['no', 'n']:
                print("Skipping convergence test.")
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

        # Kepler circular validation
        while True:
            kepler_input = input("\nDo you want to run the Kepler validation for a circular orbit? (yes/no): ").strip().lower()
            if kepler_input in ['yes', 'y']:
                print("Running Kepler validation...")
                plot_kepler_validation(Kepler_validation(params))
                break
            elif kepler_input in ['no', 'n']:
                print("Skipping Circular Kepler validation.")
                break
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

        while True:
            another = input("\nDo you want to select another orbit from the parameters file? (yes/no): ").strip().lower()
            if another in ['yes', 'y']:
                break
            elif another in ['no', 'n']:
                print("Program complete.")
                exit()
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")


        print("\nProgram complete.")