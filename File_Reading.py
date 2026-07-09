import numpy as np

Integration_methods = ['RK45', 'RK23', 'Radau', 'LSODA', 'BDF', 'DOP853']

def read_parameters(filename = "parameters.txt"):
    """Read in the parameters for the geodesic integration from a text file."""
    
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
        
        # Parse header line
        headers = [h.strip() for h in lines[0].split(',')]
        
        # Parse data line (first data row)
        if len(lines) < 2:
            raise ValueError("File must contain header and at least one line of data.")
        
        all_orbits = []
        for line_num in range(1, len(lines)):
            line = lines[line_num].strip()
            if not line:
                continue  # Skip empty lines
        
            values = [v.strip() for v in line.split(',')]

            parameters = {}

            for i in range(len(headers)):
                    if i >= len(values) or values[i] == '':
                        continue
                    try:
                        parameters[headers[i]] = float(values[i])
                    except ValueError:
                        parameters[headers[i]] = values[i]
            
            all_orbits.append(parameters)

    except FileNotFoundError:
        raise FileNotFoundError(f"Parameter file '{filename}' not found.")
    
    return all_orbits

def select_orbit(all_orbits):
    """Display available orbits and let the user select one by number."""
    
    print("\nAvailable orbits:")
    for i, orbit in enumerate(all_orbits):
        name = orbit.get('name', 'Unnamed')
        print(f"  {i + 1}. {name}")
    
    while True:
        choice = input(f"\nSelect an orbit (1-{len(all_orbits)}): ").strip()
        if choice in ['no', 'n']:
            return None
        try:
            index = int(choice) - 1
            if 0 <= index < len(all_orbits):
                print(f"Selected: {all_orbits[index].get('name', 'Unnamed')}")
                return all_orbits[index]
            else:
                print(f"Please enter a number between 1 and {len(all_orbits)}.")
        except ValueError:
            print("Invalid input. Please enter a number or 'no' to exit.")

def validation(parameters):
    """Run checks to ensure all parameters are reasonable. Establishes which variables are required,
       and which variables are optional."""

    required = ['M', 'r_0']
    optional = {'v_r': 0.0, 'v_phi': 0.0, 'method': 'DOP853', 'max_tau': 100, 'n_eval': 1000, 'orbit_type': 'custom'} # Sets default values for the optional parameters
    
    for param in required:
        if param not in parameters:
            raise ValueError(f"Missing required parameter: {param}")
    
    # Set default values for optional parameters if not provided
    for param, default in optional.items():
        if param not in parameters:
            parameters[param] = default
            print(f"Optional variable '{param}' not provided. Using default value: {default}")

    # These checks ensure that the values are positive.
    if parameters['M'] <= 0:
        raise ValueError("Mass of the black hole must be positive.")
    
    if parameters['r_0'] <= 0:
        raise ValueError("Initial radius must be positive.")

    if parameters['v_r'] < 0:
        raise ValueError("Radial velocity cannot be negative.")
    
    if parameters['v_phi'] < 0:
        raise ValueError("Angular velocity cannot be negative.")
    
    if parameters['max_tau'] <= 0:
        raise ValueError("Maximum time (max_tau) for the integration must be positive.")
    
    # The following checks ensure that the parameters are numerical values.
    if not isinstance(parameters['M'], (int, float)):
        raise ValueError(f"Mass of the black hole must be a numerical value. Got type {type(parameters['M'])} instead.")

    if not isinstance(parameters['r_0'], (int, float)):
        raise ValueError(f"Initial radius must be a numerical value. Got type {type(parameters['r_0'])} instead.")
    
    if not isinstance(parameters['v_r'], (int, float)):
        raise ValueError(f"Radial velocity must be a numerical value. Got type {type(parameters['v_r'])} instead.")
    
    if not isinstance(parameters['v_phi'], (int, float)):
        raise ValueError(f"Angular velocity must be a numerical value. Got type {type(parameters['v_phi'])} instead.")

    if not isinstance(parameters['max_tau'], (int, float)):
        raise ValueError(f"Maximum time (max_tau) must be a numerical value. Got type {type(parameters['max_tau'])} instead.")

    # Checks if the method provided is in the list of allowed methods for solve_ivp
    if parameters['method'] not in Integration_methods:
        raise ValueError(f"Invalid integration method '{parameters['method']}'. Allowed methods are: {', '.join(Integration_methods)}.")

    # Check that orbital type is a valid option
    if parameters['orbit_type'] not in ['circular', 'elliptical', 'custom']:
            raise ValueError(f"Invalid orbit type wrote in text file '{parameters['orbit_type']}'. Allowed types are: circular, elliptical, custom.")

    if parameters['orbit_type'] == 'elliptical':
        if 'r_peri' not in parameters:
            raise ValueError("Elliptical orbit requires 'r_peri' in the parameters file.")
        if 'eccentricity' not in parameters:
            raise ValueError("Elliptical orbit requires 'eccentricity' in the parameters file.")

    return True

def write_results(trajectories, orbit_name="", filename="results.csv"):
    """Write trajectory data to a CSV file, saved in a subfolder named after the orbit."""
    
    import os
    safe_name = orbit_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    output_path = os.path.join('plots', safe_name)
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)
    
    tau = trajectories['tau']
    
    if len(tau) > 1_000_000:
        step = 1000
        print(f"Large amount of timesteps ({len(tau)} timesteps). Writing every {step}th timestep.")
    else:
        step = 1
    
    t = trajectories['t']
    r = trajectories['r']
    phi = trajectories['phi']
    v_t = trajectories['v_t']
    v_r = trajectories['v_r']
    v_phi = trajectories['v_phi']
    energy = trajectories['energy']
    ang_mom = trajectories['angular_momentum']
    invariant = trajectories['invariant']
    x = trajectories['x_cartesian']
    y = trajectories['y_cartesian']
    
    with open(filepath, 'w') as f:
        f.write("tau,t,r,phi,v_t,v_r,v_phi,energy,angular_momentum,invariant,x,y\n")
        for i in range(0, len(tau), step):
            f.write(f"{tau[i]:.10e},{t[i]:.10e},{r[i]:.10e},{phi[i]:.10e},"
                    f"{v_t[i]:.10e},{v_r[i]:.10e},{v_phi[i]:.10e},"
                    f"{energy[i]:.10e},{ang_mom[i]:.10e},{invariant[i]:.10e},"
                    f"{x[i]:.10e},{y[i]:.10e}\n")
    
    print(f"Results written to {filepath} ({len(range(0, len(tau), step))} of {len(tau)} timesteps)")
        