# This program solves the 2D Ising model using a Metropolis Monte Carlo algorithm.
import numpy as np
import random
import matplotlib.pyplot as plt
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
    energies = np.zeros((1, iterations))
    spins = np.zeros((1, iterations))
    # it = []
    
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
        E_i = calculate_total_energy(lattice, J) # energy before spin flip
        initial_spin = calculate_total_spin(lattice)
        
        # update to the new state and calculate the new energy and total spin
        new = -1*old
        lattice[i][j] = new
        E_f = calculate_total_energy(lattice, J) # energy after spin flip
        final_spin = calculate_total_spin(lattice)
        
        # determine the change in energy
        delta_E = E_f - E_i
        
        P = np.random.uniform()
        # boltzmann factor exp(-\delta E/k_B*T)
        bz = np.exp(-delta_E/(k_B*T))
        
        # determine whether to accept the new state
        if E_f < E_i:
            t += 1
            spins.append(final_spin)
            energies.append(E_f)
            # it.append(t)
            continue
        elif P < bz:
            t += 1
            spins.append(initial_spin)
            energies.append(E_f)
            # it.append(t)
            continue
        else:
            # revert to the old state and move on
            t += 1
            lattice[i][j] = old
            spins.append(initial_spin)
            energies.append(E_i)
            # it.append(t)
            continue
    energies = np.array(energies)
    spins = np.array(energies)
    
    # return the final state
    # energy = np.divide(np.sum(energies), len(energies))
    # spin = np.divide(np.sum(spins), len(spins))
    return lattice, energy, spin

def plot(lattice, i):
    plt.figure(i)
    plt.imshow(lattice)
    plt.colorbar()
    # save the file by temperature identifier
    name = "fig_" + str(50 + 10*i) + ".png"
    plt.savefig(name)
    plt.close()
    return

# Take a thermodynamic average of a particular quantity
# Q: Quantity of interest
# Z: Partition function
# energies: The energy levels of the system
# @jit(nopython=True)
def thermodynamic_average(Q, Z, energies, T):
    if Q.size != energies.size:
        print("Error: Number of samples of Q does not match the number of energy samples (levels)")
        return None
    num = np.sum(np.multiply(Q, np.exp(np.divide(-energies, (k_B*T)))))
    avg = num / Z
    return avg

N = 50
iterations = 20000

# Run for NxN lattice for temperatures between 0 and 300K in 10K intervals
temperatures = np.arange(50, 310, 5)   

tick = time.time()

energies = []
spins = []
for i in range(0, temperatures.size):
    T = str(temperatures[i]) + "K"
    print("Temperature = " + T)
    lattice, energy, spin = metropolis(initial_state(N,N), iterations, temperatures[i])
    energies.append(energy)
    spins.append(spin)
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
