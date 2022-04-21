import numpy as np
import matplotlib.pyplot as plt
from metro import metropolis, initial_state
import matplotlib.colors as clr
from datetime import datetime
import time

# Begin adjustable parameters
T_i = 300
T_f = 340
step = 0.1

# I_eq: iterations required to reach the equilibrium state
I_eq = 10000
# I_s: iterations over which the thermodynamic properties
# are calculated.
I_s = 100000
# dimension of lattice. In this case we are using a square lattice,
# though this code accounts for rectangular lattices also.
n = 20

# Make a plot of the spins
def plot(lattice, T):
    cmap = clr.ListedColormap(['black', 'white'])

    plt.figure()
    plt.imshow(lattice, cmap)

    # save the file by temperature identifier
    name = "fig_" + str(T) + ".png"
    plt.xlabel("T = " + str(T))
    plt.savefig(name)
    plt.close()
    return

def make_plots(data, T_c):
    dt = datetime.fromtimestamp(time.time())
    
    temperatures = data[0]
    energies = data[1]
    spins = data[2]
    heat_caps = data[3]
    susceptibilities = data[4]
    
    plt.figure()
    plt.plot(temperatures, energies, c="red")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Average energy (J)")
    plt.savefig("energy_vs_temperature_" + str(dt) + ".png")

    plt.figure()
    plt.scatter(temperatures, spins, c="red")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Magnetisation")
    plt.savefig("magnetisation_vs_temperature_" + str(dt) + ".png")

    plt.figure()
    plt.plot(temperatures, susceptibilities, c="red")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Susceptibility")
    plt.savefig("susceptibility_vs_temperature_" + str(dt) + ".png")

    plt.figure()
    plt.plot(temperatures, heat_caps, c="red")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Heat capacity")
    plt.savefig("heatcap_vs_temperature_" + str(dt) + ".png")
 
    return

def ising():
    print("ising: Implements 2D Ising model. To play around with the parameters used in this model, see `ising.py`.")
    # Prepare data
    temperatures = np.arange(T_i, T_f, step)
    energies = np.zeros((temperatures.size,))
    spins = np.zeros((temperatures.size,))
    heat_caps = np.zeros((temperatures.size,))
    susceptibilities = np.zeros((temperatures.size,))
    
    temp_vals = np.arange(0, 400, 10)

    print("ising: Beginning calculation of thermodynamic properties.")
    for i in range(0, temperatures.size):
        T = temperatures[i]

        print("Temperature = " + str(T) + "K.")
        
        # First solve for the equilibrium state. 
        eq_state = metropolis(initial_state(n,n), I_eq, T)[0]
        # Now calculate statistics 
        data = metropolis(eq_state, I_s, T)[1] 

        if T in temp_vals:
            plot(eq_state, T)

        energies[i] = data[0]
        spins[i] = data[1]
        heat_caps[i] = data[2]
        susceptibilities[i] = data[3]    


    T_c = temperatures[phase_transition(susceptibilities)]
    print("The phase transition occurs at a temperature of " + str(T_c) + " K")

    data = np.array([temperatures, energies, spins, heat_caps, susceptibilities])
    write_data(data)
    make_plots(data, T_c)

    return

def write_data(data):
    dt = datetime.fromtimestamp(time.time())
    f = "ising_data_" + str(dt) + ".dat"

    with open(f, 'w') as dat:
        heading = "T    E    M    C    X\n"
        dat.write(heading)
        
        for i in range(0, data[0].size):
            line = str(data[0][i]) + "    " + str(data[1][i]) + "    " + str(data[2][i]) + "    " + str(data[3][i]) + "    " + str(data[4][i]) + "\n"
            dat.write(line)
        dat.close()


def phase_transition(suscept_data):
    suscept_i = suscept_data[0:(suscept_data.size - 1)]
    # roll back each element by 1 index, so that
    # we can subtract elements pairwise.
    suscept_f = np.roll(suscept_data, -1)[0:(suscept_data.size - 1)]
    gradients = np.zeros(suscept_i.shape)
    
    # Search for a sudden large increase in the susceptibility,
    # corresponding to the phase transition
    for i in range(0, suscept_i.size):
        gradient = suscept_f[i] - suscept_i[i]
        gradients[i] = gradient
    j = np.where(gradients == np.amax(gradients))
    return j


def make_iteration_plots():
    it = np.arange(0, 30000)

    raw_energies = metropolis(initial_state(n,n), 30000, 10)[2]
    plt.plot(it, raw_energies, c="purple")
    plt.xlabel("Iterations")
    plt.ylabel("Lattice energy")
    plt.savefig("iterations_vs_temperature_n=" + str(n))

ising()
