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
# Exchange energy
J = np.power(10.0, -21)

# Adjustable parameters that govern the scale of the simulation
N = 20


# Since there are only four nearest neighbours,
# there are a limited number of ways the spins can be arranged/oriented
# that give the same energy for a given spin. These means there are also a limited
# number of possible values for the change in energy due to the spin flip
def acceptor_probabilities(T):
    probs = {0: 1,
             2: np.exp(-2*J/(k_B*T)),
             -2: np.exp(2*J/(k_B*T)),
             4: np.exp(-4*J/(k_B*T)),
             -4: np.exp(4*J/(k_B*T)),
             6: np.exp(-6*J/(k_B*T)),
             -6: np.exp(6*J/(k_B*T)),
             8: np.exp(-8*J/(k_B*T)),
             -8: np.exp(8*J/(k_B*T))
            }
    return probs


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


# i: 0 -> x - 1
# j: 0 -> y - 1
@jit(nopython=True)
def nearest_neighbours(lattice, i, j):
    (x,y) = lattice.shape

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
    # Total lattice sites
    n = x*y

    energies = np.zeros((iterations,))
    spins = np.zeros((iterations,))
    energies_sq = np.zeros((iterations,))
    spins_sq = np.zeros((iterations,))

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
        
        # determine whether to accept the new state
        if delta_E <= 0:
            energies[t] = E_f
            spins[t] = S_f
            energies_sq[t] = E_2f
            spins_sq[t] = S_2f
            t += 1

            continue
        elif P < np.exp(-delta_E*J/(k_B*T)):
            energies[t] = E_f
            spins[t] = S_f
            energies_sq[t] = E_2f
            spins_sq[t] = S_2f
            t += 1

            continue
        else:
            # revert to the old state and move on
            lattice[i][j] = old
            energies[t] = E_i
            spins[t] = S_i
            energies_sq[t] = E_2i
            spins_sq[t] = S_2i
            t += 1

            continue
    
    energy = np.sum(energies)/iterations
    magnetisation = np.sum(spins)/iterations
    heat_cap = np.divide((np.sum(energies_sq)/iterations - energy**2), (k_B*(T**2)))
    suscept = np.divide((np.sum(spins_sq)/iterations - magnetisation**2), (k_B*T))

    # return the final state
    return lattice, np.array([energy, magnetisation, heat_cap, suscept])

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
