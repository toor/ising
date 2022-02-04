# This program solves the 2D Ising model using a Metropolis Monte Carlo algorithm.
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# used for runtime stats
import time
from numba import jit

# Boltzmann's constant
k_B = 1.38*np.power(10.0, -23)

# Adjustable parameters that govern the scale of the simulation
N = 20
iterations = 20000


@jit(nopython=True)
def initial_state(x, y):
    # 75% negative
    p = 0.25
    # return array of random floats between 0,1 with the same shape as
    # the lattice
    rand = np.random.random((x,y))
    lattice = np.zeros((x,y))
    for i in range(0, x):
        for j in range(0,y):
            # p% of the spins up
            if rand[i][j] >= p:
                lattice[i][j] = 1
            # (1 - p)% of the spins down
            else:
                lattice[i][j] = -1
    
    return lattice


@jit(nopython=True)
def nearest_neighbours(lattice, i, j):
    (x,y) = lattice.shape
    x -= 1
    y -= 1
    nn = [lattice[(i - 1)%x][j], lattice[(i+1)%x][j], lattice[i][(j-1)%y], lattice[i][(j + 1)%y]]

    nn = np.array(nn)

    return nn

# Calculate the total energy by iterating over all lattice sites (spins) and calculating their contribution
# to the exchange energy. The exchange interaction is assumed to be limited to nearest neighbours.
@jit(nopython=True)
def calculate_total_energy(lattice):
    energy = 0
    # rows, cols
    (x, y) = lattice.shape
    # iterate over rows
    for i in range(0, x):
        # iterate over columns
        for j in range(0, y):
            spin = lattice[i][j]
            nn = nearest_neighbours(lattice, i, j)
            for neighbour in nn:
                contribution = -spin*neighbour
                energy += contribution
   
    return energy

# Determine the total spin-state of the lattice for a 
# particular config.
@jit(nopython=True)
def calculate_total_spin(lattice):
    spin = 0
    (x,y) = lattice.shape
    for i in range(0,x):
        for j in range(0,y):
            _spin = lattice[i][j]
            spin += _spin
    return spin

@jit(nopython=True)
# Returns the total average energy per spin, average spin,
# average energy squared and average spin squared
def energy_and_spin(lattice, n):
    E_total = calculate_total_energy(lattice)
    E_2_total = np.power(E_total, 2)
    E = np.divide(E_total, n)
    E_2 = np.divide(E_2_total, n)

    S_total = calculate_total_spin(lattice)
    S_2_total = np.power(S_total, 2)
    S = np.divide(S_total, n)
    S_2 = np.divide(S_2_total, n) 

    return E, E_2, S, S_2, E_total


# Run metropolis on a specific lattice for a given temperature
def metropolis(lattice, iterations, T):
    (x,y) = lattice.shape
    n = x*y
    energy = 0
    # squared energy; will be used to plot the heat capacity
    enrg_sq = 0
    magnetisation = 0
    mag_sq = 0


    # Choose an interaction energy that is of the same order as the thermal energy kT ~10^-21 for
    # T between 0 and 1000K
    J = np.power(10.0, -21)
    
    t = 0  
    # Run a specified number of Monte Carlo steps
    while t < iterations:
        # return i in the interval [0, x - 1]
        i = np.random.randint(x)
        j = np.random.randint(y)
       
        # calculate the total energy in units of J, and thus
        # the average energy per spin 
        old = lattice[i][j]
        E_i, E_2i, S_i, S_2i, E_i_total = energy_and_spin(lattice, n)

        # update to the new state
        new = -1*old
        lattice[i][j] = new
    
        E_f, E_2f, S_f, S_2f, E_f_total = energy_and_spin(lattice, n)

        # determine the change in energy
        delta_E = E_f_total - E_i_total
        
        P = np.random.uniform()
        # boltzmann factor exp(-\delta E/k_B*T)
        bz = np.exp(-delta_E*J/(k_B*T))
        
        # determine whether to accept the new state
        if delta_E < 0:
            energy += E_f
            magnetisation += S_f
            enrg_sq += E_2f
            mag_sq += S_2f
            t += 1

            continue
        elif P < bz:
            energy += E_f
            magnetisation += S_f
            enrg_sq += E_2f
            mag_sq += S_2f
            t += 1

            continue
        else:
            # revert to the old state and move on
            lattice[i][j] = old
            energy += E_i
            magnetisation += S_i
            enrg_sq += E_2i
            mag_sq += S_2i
            t += 1

            continue

    energy = energy/iterations
    magnetisation = magnetisation/iterations
    heat_cap = np.divide((enrg_sq - energy**2), (k_B*(T**2)))/iterations
    suscept = np.divide((mag_sq - magnetisation**2), (k_B*T))/iterations

    # return the final state
    return lattice, energy, magnetisation, heat_cap, suscept

# Make a plot of the spins
def plot(lattice, i, T):
    cmap = colors.ListedColormap(['red', 'blue'])

    plt.figure(i)
    plt.imshow(lattice, cmap)

    # save the file by temperature identifier
    name = "fig_" + str(i) + ".png"
    plt.xlabel("T = " + str(T))
    plt.savefig(name)
    plt.close()
    return


# Run for NxN lattice for temperatures between 0 and 300K in 10K intervals
temperatures = np.arange(1, 310, 1)   

tick = time.time()

energies = np.zeros((temperatures.size,))
spins = np.zeros((temperatures.size,))
heat_caps = np.zeros((temperatures.size,))
susceptibilities = np.zeros((temperatures.size,))

for i in range(0, temperatures.size):
    print("Temperature = " + str(temperatures[i]))
    lattice, energy, spin, heat, suscept = metropolis(initial_state(N,N), iterations, temperatures[i])
    energies[i] = energy
    spins[i] = spin
    heat_caps[i] = heat
    susceptibilities[i] = suscept

    #plot(lattice, i, T)

plt.figure()
plt.scatter(temperatures, energies)
plt.xlabel("Temperature (K)")
plt.ylabel("Average energy E/J")
plt.savefig("energy_vs_temperature.png")
plt.close()

plt.figure()
plt.scatter(temperatures, spins)
plt.xlabel("Temperature (K)")
plt.ylabel("Magnetisation")
plt.savefig("magnetisation_vs_temperature.png")

plt.figure()
plt.scatter(temperatures, heat_caps)
plt.xlabel("Temperature (K)")
plt.ylabel("Heat capacity (Joules/KJ^2 )")
plt.savefig("specific_heat_vs_temperature.png")

plt.figure()
plt.scatter(temperatures, susceptibilities)
plt.xlabel("Temperature (K)")
plt.ylabel("Susceptibility")
plt.savefig("susceptibility_vs_temperature.png")

# Measure time taken
tock = time.time()
duration = tock - tick
print(duration)
