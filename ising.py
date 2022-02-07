import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from ising_metropolis import metropolis, initial_state

T_i = 1
T_f = 300
step = 1

# I_eq: iterations required to reach the equilibrium state
I_eq = 2500
# I_s: iterations over which the thermodynamic properties
# are calculated.
I_s = 20000
# dimension of lattice. In this case we are using a square lattice,
# though this code accounts for rectangular lattices also.
n = 20

def magnetisation_T(x, T_c, gamma, A):
    # Theoretically, T_c is ~164K.
    return A*np.power((T_c - x), gamma)

def make_plots(data):
    temperatures = data[0]
    energies = data[1]
    spins = data[2]
    heat_caps = data[3]
    susceptibilities = data[4]

    # Fit a curve to the magnetisation. We expect the magnetisation to go
    # like M ~ (T_c - T)^{\gamma} where \gamma is a parameter to fit - see
    # magnetisation_T.
    print("plots: attempting to fit a curve to magnetisation data")
    popt,pcov = curve_fit(magnetisation_T, temperatures, spins, bounds=([295, 0, 0.1], [305, 1, 1]))

    print(popt)
    _mag = np.zeros((temperatures.size,))
    for j in range(0, temperatures.size):
        _mag[j] = magnetisation_T(temperatures[j], popt[0], popt[1], popt[2])
        
    plt.figure()
    plt.scatter(temperatures, energies, c="red", marker='x')
    plt.xlabel("Temperature (K)")
    plt.ylabel("Average energy E/J")
    plt.savefig("energy_vs_temperature.png")

    plt.figure()
    plt.scatter(temperatures, spins, c="red", marker='x')
    plt.plot(temperatures, _mag, c="blue")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Magnetisation")
    plt.savefig("magnetisation_vs_temperature.png")

    plt.figure()
    plt.scatter(temperatures, heat_caps, c="red", marker='x')
    plt.xlabel("Temperature (K)")
    plt.ylabel("Heat capacity C_V (JK^-1)")
    plt.savefig("heat_capacity_vs_temperature.png")

    plt.figure()
    plt.scatter(temperatures, susceptibilities, c="red", marker='x')
    plt.xlabel("Temperature (K)")
    plt.ylabel("Susceptibility")
    plt.savefig("susceptibility_vs_temperature.png")
    
    return

# TODO: Make initial, final temperatures, iterations and lattice size CLI parameters
# note however that the number of iterations required for "thermalisation" - the period
# over which the system approaches thermal equilibrium - will depend on the dimensions
# of the lattice in question.
def ising():
    temperatures = np.arange(T_i, T_f, step)

    # Prepare data
    energies = np.zeros((temperatures.size,))
    spins = np.zeros((temperatures.size,))
    heat_caps = np.zeros((temperatures.size,))
    susceptibilities = np.zeros((temperatures.size,))

    for i in range(0, temperatures.size):
        print("Temperature = " + str(temperatures[i]) + "K")
        print("Running Metropolis algorithm to allow system to equilibrate for " + str(I_eq) + " iterations.")
        # First solve for the equilibrium state. For a 20x20 lattice, 2500 iterations are sufficient
        # remove ^
        eq_state = metropolis(initial_state(n,n), I_eq, temperatures[i])[0]
        print("Now calculating thermodynamic properties over " + str(I_s) + " iterations.")
        lattice, data = metropolis(eq_state, I_s, temperatures[i])
        energies[i] = data[0]
        spins[i] = data[1]
        heat_caps[i] = data[2]
        susceptibilities[i] = data[3]

    data = np.array([temperatures, energies, spins, heat_caps, susceptibilities]) 
    make_plots(data)


ising()
