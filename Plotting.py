import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def _save_plot(figure, filename, orbit_name=""):
    """Save a matplotlib figure to a subfolder named after the orbit."""
    safe_name = orbit_name.replace(" ", "_").replace("/", "_").replace("\\", "_") # Make a folder name
    output_path = os.path.join('plots', safe_name)
    os.makedirs(output_path, exist_ok=True) # Create the folder if it doesn't exist
    figure.savefig(os.path.join(output_path, filename), dpi=300)

def plot_trajectories(trajectories, name = ""):
    """Plot the trajectories in Cartesian coordinates."""
    
    x = trajectories['x_cartesian']
    y = trajectories['y_cartesian']

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Trajectory', color='blue')
    
    # Plot the black hole as a circle at the origin
    R_s = trajectories['R_s']
    black_hole = Circle((0, 0), R_s, color='black', label='Black Hole')
    plt.gca().add_patch(black_hole)

    # ISCO at 3 R_s
    isco = Circle((0, 0), 3 * R_s, color='grey', fill=False, linestyle='--', label='ISCO (3 $R_s$)')
    plt.gca().add_patch(isco)

    # Green dot at start
    plt.scatter(x[0], y[0], color='green', s=50, label='Start', zorder=5)
    
    # Red dot at end
    plt.scatter(x[-1], y[-1], color='red', s=50, label='End', zorder=5)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Particle Trajectory for a {name}')
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()

    _save_plot(plt.gcf(), "Trajectory.png", name)
    print(f'Plot complete, saved to: plots/{name.replace(" ", "_")}/{name}_trajectory.png')
    plt.close()

def plot_energy_conservation(trajectories,  name=""):
    """Plot specific energy vs proper time and the corresponding residual."""
    
    tau = trajectories['tau']
    energy = trajectories['energy']
    residual_energy = energy - energy[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 6))
    fig.suptitle('Energy Conservation and Residual', fontsize=16)
    
    ax1.plot(tau, energy, color='blue')
    ax1.axhline(energy[0], color='red', linestyle='--', alpha=0.5, label=f'Initial = {energy[0]:.6f}')
    ax1.set_ylabel('Specific Energy')
    ax1.set_xlabel(r'Proper Time $\tau$')
    ax1.set_title('Energy Conservation')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.ticklabel_format(useOffset=False, axis='y')

    ax2.plot(tau, residual_energy, color='orange')
    ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel(r'$\Delta$ Specific Energy')
    ax2.set_xlabel(r'Proper Time $\tau$')
    ax2.set_title('Energy Residual')
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    _save_plot(plt.gcf(), "Energy_conservation.png", name)
    print(f'Plot complete, saved to: plots/{name.replace(" ", "_")}/{name}_energy.png')
    plt.close()

def plot_ang_mom_conservation(trajectories, name=""):
    """Plot specific angular momentum vs proper time and the corresponding residual."""
    
    tau = trajectories['tau']
    ang_mom = trajectories['angular_momentum']
    residual_ang_mom = ang_mom - ang_mom[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 6))
    fig.suptitle('Angular Momentum Conservation and Residual', fontsize=16)
    
    ax1.plot(tau, ang_mom, color='green')
    ax1.axhline(ang_mom[0], color='red', linestyle='--', alpha=0.5, label=f'Initial = {ang_mom[0]:.6f}')
    ax1.set_ylabel('Specific Angular Momentum')
    ax1.set_xlabel(r'Proper Time $\tau$')
    ax1.set_title('Angular Momentum Conservation')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.ticklabel_format(useOffset=False, axis='y')

    ax2.plot(tau, residual_ang_mom, color='orange')
    ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel(r'$\Delta$ Specific Angular Momentum')
    ax2.set_xlabel(r'Proper Time $\tau$')
    ax2.set_title('Angular Momentum Residual')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()

    _save_plot(plt.gcf(), "Angular_momentum_conservation.png", name)
    print(f'Plot complete, saved to: plots/{name.replace(" ", "_")}/{name}_angular_momentum.png')
    plt.close()

def plot_invariant_conservation(trajectories, name=""):
    """Plot metric invariant vs proper time and the corresponding residual."""
    
    tau = trajectories['tau']
    invariant = trajectories['invariant']
    residual_invariant = invariant - invariant[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 6))
    fig.suptitle('Metric Invariant Conservation and Residual', fontsize=16)
    
    ax1.plot(tau, invariant, color='purple')
    ax1.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Expected = 1')
    ax1.set_ylabel('Invariant')
    ax1.set_xlabel(r'Proper Time $\tau$')
    ax1.set_title('Metric Invariant Conservation')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.ticklabel_format(useOffset=False, axis='y')

    ax2.plot(tau, residual_invariant, color='orange')
    ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel(r'$\Delta$ Metric Invariant')
    ax2.set_xlabel(r'Proper Time $\tau$')
    ax2.set_title('Metric Invariant Residual')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()

    _save_plot(plt.gcf(), "Invariant_conservation.png", name)
    print(f'Plot complete, saved to: plots/{name.replace(" ", "_")}/{name}_invariant.png')
    plt.close()

def plot_convergence(convergence, name=""):
    """Log-log plot of conservation errors vs integration tolerance."""
    
    tols = np.array(convergence['tolerances'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(tols, convergence['invariant_error'], 'bo-')
    
    ax.set_xlabel('Tolerance (rtol = atol)', fontsize=14)
    ax.set_ylabel('Max absolute Invariant error', fontsize=14)
    ax.set_title('Convergence Test', fontsize=16)
    ax.grid(True, which='both', alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()

    _save_plot(plt.gcf(), "Convergence_test.png", name)
    print(f'Plot complete, saved to: plots/{name.replace(" ", "_")}/{name}_convergence.png')
    plt.close()

def plot_time_dilation(trajectories, name=""):
    """Plot coordinate time vs proper time, showing gravitational time dilation."""
    
    tau = trajectories['tau']
    t_coord = trajectories['t']
    
    # Coordinate time vs proper time
    plt.figure(figsize=(10, 6))
    plt.plot(tau, t_coord, color='blue')
    plt.plot(tau, tau, 'r--', alpha=0.5, label='No time dilation')
    plt.xlabel(r'Proper Time $\tau$')
    plt.ylabel(r'Coordinate Time $t$')
    plt.title('Coordinate Time vs Proper Time')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()

    _save_plot(plt.gcf(), "Time_dilation.png", name)
    print(f'Plot complete, saved to: plots/{name.replace(" ", "_")}/{name}_time_dilation.png')
    plt.close()

def plot_kepler_validation(Kepler):
    """Plot the results of the Kepler validation to show T^2 vs a^3 for different values of a."""

    a_values = np.array(Kepler['radii'])
    expected_periods = np.array(Kepler['expected_periods'])
    measured_periods = np.array(Kepler['measured_periods'])
    residuals = np.array(Kepler['residuals'])

    # Calculate T^2 and a^3 for plotting
    T_squared_measured = measured_periods ** 2
    T_squared_expected = expected_periods ** 2
    a_cubed = (a_values) ** 3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    ax1.plot(a_cubed, T_squared_measured, 'bo', markersize=8, label='Measured')
    ax1.plot(a_cubed, T_squared_expected, 'r--', linewidth=2, label='Expected')
    ax1.set_xlabel(r'$r^3$')
    ax1.set_ylabel(r'$T^2$')
    ax1.set_title("Kepler's Third Law Validation")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.scatter(a_cubed, residuals, s=50)
    ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel(r'$r^3$')
    ax2.set_ylabel(r'$Percentage Residual$')
    ax2.set_title("Kepler Residuals")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    fig.savefig(os.path.join('plots', 'kepler_validation.png'), dpi=300)
    plt.close()
    print('Plot complete, saved to: plots/kepler_validation.png')