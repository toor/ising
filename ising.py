import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from ising_metropolis import metropolis, initial_state
import matplotlib.colors as clr


T_i = 1
T_f = 305
step = 1

# I_eq: iterations required to reach the equilibrium state
I_eq = 2500
# I_s: iterations over which the thermodynamic properties
# are calculated.
I_s = 10000
# dimension of lattice. In this case we are using a square lattice,
# though this code accounts for rectangular lattices also.
n = 20

def magnetisation_T(x, T_c, gamma, A):
    # Theoretically, T_c is ~164K.
    return A*np.power((T_c - x), gamma)

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
    plt.ylabel("Average energy E/J")
    plt.savefig("energy_vs_temperature.png")

    plt.figure()
    plt.plot(temperatures, spins, c="red")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Magnetisation")
    plt.savefig("magnetisation_vs_temperature.png")

    plt.figure()
    plt.plot(temperatures, heat_caps, c="red")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Heat capacity C_V (JK^-1)")
    plt.savefig("heat_capacity_vs_temperature.png")

    plt.figure()
    plt.plot(temperatures, susceptibilities, c="red")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Susceptibility")
    plt.savefig("susceptibility_vs_temperature.png")
    
    return

# spin_gradient: Determines the slope between each pair of points
# in the spin vs. temperature distribution.
# Parameters: temperatures - array of temperatures (ndarray.int64)
#             spins - array of spins (ndarray.float64)
# Returns: slopes - array of slopes (ndarray.float64)
def spin_gradient(temperatures, spins):
    spins_i = spins[0:(temperatures.size - 1)]
    spins_f = np.roll(spins, -1)[0:(temperatures.size - 1)]
    temps_i = temperatures[0:(temperatures.size - 1)]
    temps_f = np.roll(temperatures, - 1)[0:(temperatures.size - 1)]

    delta_S = np.subtract(spins_f, spins_i)
    delta_T = np.subtract(temps_f, temps_i)

    slopes = np.divide(delta_S, delta_T)

    return slopes


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

    special_temps = np.array([50, 100, 150, 200, 250, 300])

    for i in range(0, temperatures.size):
        T = temperatures[i]

        print("Temperature = " + str(T) + "K")
        
        print("Running Metropolis algorithm to allow system to equilibrate for " + str(I_eq) + " iterations.")
        # First solve for the equilibrium state. For a 20x20 lattice, 2500 iterations are sufficient
        # remove ^
        eq_state = metropolis(initial_state(n,n), I_eq, T)[0]
        
        print("Now calculating thermodynamic properties over " + str(I_s) + " iterations.")
        lattice, data = metropolis(eq_state, I_s, T)
        
        if np.isin(T, special_temps):
            print("Plotting lattice for T = " + str(T) + "K")
            plot(eq_state, T)

        energies[i] = data[0]
        spins[i] = data[1]
        heat_caps[i] = data[2]
        susceptibilities[i] = data[3]
    
    # make an array that will contain tuples of the slopes,
    # together with the temperatures they correspond to. First define a new datatype that will
    # allow us to sort the resulting array
    _type = [('slope', float), ('T_i', float), ('T_f', float)]

    # Iterate pairwise over temperatures and spins and determine the slope.
    # TODO: Is there a way to do this without explicity iterating over
    # `temperatures`? e.g. create another array of temperatures, offset by 1,
    # and could then use np.subtract to subtract elements pairwise and then could simply
    # read off the slopes from an array
    # example: temperatures = [0,1,2,3,4]
    #       => temperatures' = [1,2,3,4, None] since we want to neglect the last element
    slopes = spin_gradient(temperatures, spins)
    # This is a bit ugly, but I don't really know how to zip up these arrays
    # correctly
    values = np.array([(slopes[i],
        temperatures[i],
        temperatures[i + 1]) for i in range(0, temperatures.size - 1)], dtype=_type)

    # This is pretty slow - the complexity of numpy's sort can be as bad
    # as O(n*log(n)). Anyway, this sorts `values` - recall these elements are tuples - according
    # to the slope
    np.sort(values, order='slope')
    print(values)

    data = np.array([temperatures, energies, spins, heat_caps, susceptibilities]) 
    make_plots(data)


ising()
