# This program solves the 2D Ising model using a Metropolis Monte Carlo algorithm.
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors
# used for runtime stats
import time
from numba import jit

# Boltzmann's constant
k_B = 1.38*np.power(10.0, -23)

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
def calculate_total_energy(lattice, J):
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
                contribution = -J*spin*neighbour
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


# Run metropolis on a specific lattice for a given temperature
def metropolis(lattice, iterations, T):
    (x, y) = lattice.shape
    n = x*y
    energies = np.zeros((iterations,))
    spins = np.zeros((iterations,))
    
    # Choose an interaction energy that is of the same order as the thermal energy kT ~10^-21 for
    # T between 0 and 1000K
    J = 1.0*np.power(10.0, -21)
    t = 0
    (x,y) = lattice.shape
    
    # Run a specified number of Monte Carlo steps
    while t < iterations:
        # return i in the interval [0, x - 1]
        i = np.random.randint(x)
        j = np.random.randint(y)
        
        old = lattice[i][j]
        E_i_total = calculate_total_energy(lattice, J)
        E_i = np.divide(E_i_total, n)
        initial_spin = calculate_total_spin(lattice)
        
        # update to the new state and calculate the new energy and total spin
        new = -1*old
        lattice[i][j] = new
        E_f_total = calculate_total_energy(lattice, J)
        E_f = np.divide(E_f_total, n)
        final_spin = calculate_total_spin(lattice)
        
        # determine the change in energy
        delta_E = E_f - E_i
        
        P = np.random.uniform()
        # boltzmann factor exp(-\delta E/k_B*T)
        bz = np.exp(-delta_E/(k_B*T))
        
        # determine whether to accept the new state
        if E_f < E_i:
            t += 1
            np.append(spins, final_spin)
            np.append(energies, E_f/n)
            # it.append(t)
            continue
        elif P < bz:
            t += 1
            np.append(spins, final_spin)
            np.append(energies, E_f)
            # it.append(t)
            continue
        else:
            # revert to the old state and move on
            t += 1
            lattice[i][j] = old
            np.append(spins, initial_spin)
            np.append(energies, E_i)
            # it.append(t)
            continue
    # Compute energy and magnetisation for this temperature
    energy = np.divide(np.sum(energies), iterations)
    spin = np.divide(np.sum(spins), iterations)

    # return the final state
    # energy = np.divide(np.sum(energies), len(energies))
    # spin = np.divide(np.sum(spins), len(spins))
    return lattice, energy, spin

# Make a plot of the spins
def plot(lattice, i):
    cmap = colors.ListedColormap(['red', 'blue'])

    plt.figure(i)
    plt.imshow(lattice)
    plt.colorbar()
    # save the file by temperature identifier
    name = "fig_" + str(50 + 10*i) + ".png"
    plt.savefig(name)
    plt.close()
    return

N = 20
iterations = 20000

# Run for NxN lattice for temperatures between 0 and 300K in 10K intervals
temperatures = np.arange(50, 310, 5)   

tick = time.time()

energies = np.zeros((temperatures.size,))
spins = np.zeros((temperatures.size,))

for i in range(0, temperatures.size):
    T = str(temperatures[i]) + "K"
    print("Temperature = " + T)
    lattice, energy, spin = metropolis(initial_state(N,N), iterations, temperatures[i])
    np.append(energies, energy)
    np.append(spins, spin)
    plot(lattice, i)

energies = np.array(energies)
spins = np.array(spins)

plt.figure()
plt.scatter(temperatures, energies)
plt.xlabel("Temperature (K)")
plt.ylabel("Lattice energy")
plt.savefig("energy_vs_temperature.png")
plt.close()

plt.figure()
plt.scatter(temperatures, spins)
plt.xlabel("Temperature (K)")
plt.ylabel("Magnetisation")
plt.savefig("magnetisation_vs_temperature.png")

# Measure time taken
tock = time.time()
duration = tock - tick
print(duration)
