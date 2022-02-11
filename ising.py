import numpy as np
import matplotlib.pyplot as plt
from metro import metropolis, initial_state
import matplotlib.colors as clr

# Begin adjustable parameters

T_i = 1
T_f = 305
step = 1

# I_eq: iterations required to reach the equilibrium state
I_eq = 3000
# I_s: iterations over which the thermodynamic properties
# are calculated.
I_s = 15000
# dimension of lattice. In this case we are using a square lattice,
# though this code accounts for rectangular lattices also.
n = 20

# Make a plot of the spins
def plot(lattice, T):
    cmap = clr.ListedColormap(['red', 'blue'])

    plt.figure()
    plt.imshow(lattice, cmap)

    # save the file by temperature identifier
    name = "fig_" + str(T) + "K" + ".png"
    plt.xlabel("T = " + str(T))
    plt.savefig(name)
    plt.close()
    return

def make_plots(data):
    temperatures = data[0]
    energies = data[1]
    spins = data[2]
    heat_caps = data[3]
    susceptibilities = data[4]
    
    plt.figure()
    plt.plot(temperatures, energies, c="red")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Average energy (J)")
    plt.savefig("energy_vs_temperature.png")

    plt.figure()
    plt.plot(temperatures, spins, c="red")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Magnetisation")
    plt.savefig("magnetisation_vs_temperature.png")

    plt.figure()
    plt.plot(temperatures, heat_caps, c="red")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Heat capacity (JK^-1)")
    plt.savefig("heat_capacity_vs_temperature.png")

    plt.figure()
    plt.plot(temperatures, susceptibilities, c="red")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Susceptibility")
    plt.savefig("susceptibility_vs_temperature.png")
    
    return

def ising():
    print("ising: Implements 2D Ising model. To play around with the parameters used in this model, see `ising.py`.")
    # Prepare data
    temperatures = np.arange(T_i, T_f, step)
    energies = np.zeros((temperatures.size,))
    spins = np.zeros((temperatures.size,))
    heat_caps = np.zeros((temperatures.size,))
    susceptibilities = np.zeros((temperatures.size,))
    
    print("ising: Beginning calculation of thermodynamic properties.")
    for i in range(0, temperatures.size):
        T = temperatures[i]

        print("Temperature = " + str(T) + "K.")
        
        # First solve for the equilibrium state. 
        eq_state = metropolis(initial_state(n,n), I_eq, T)[0]
        # Now calculate statistics 
        data = metropolis(eq_state, I_s, T)[1]

        energies[i] = data[0]
        spins[i] = data[1]
        heat_caps[i] = data[2]
        susceptibilities[i] = data[3]
    
    spins_i = spins[0:(temperatures.size - 1)]
    spins_f = np.roll(spins, -1)[0:(temperatures.size - 1)]   
    sz = spins_i.size
    s_max = np.amax(spins)
    s_min = np.min(spins)
    
    print("ising: Attempting to determine the critical temperature.")
    for j in range(0, sz - 1):
        slope1 = spins_f[j] - spins_i[j]
        slope2 = spins_f[j + 1] - spins_i[j + 1]
        if (slope2 - slope1) < 0 and np.abs((slope2 - slope1)) >= 3*(s_max - s_min)/4:
            T_c = temperatures[j]
            print("ising: The critical temperature is ~ " + str(T_c) + "K.")

    data = np.array([temperatures, energies, spins, heat_caps, susceptibilities]) 
    make_plots(data)

ising()
