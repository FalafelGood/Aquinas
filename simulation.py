import numpy as np
import sys
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from boson_sampling_probabilities import output_probability

sys.path.append('../') # Add parent directory to the system path
from direct_decomposition import direct_decomposition


def dist_to_state(dist):
    """
    Given a photon distribution, returns the corresponding binary state

    Example:
    dist := [1,3] (One photon in first mode, 3 photons in second)
    >> '001 011'
    (Note, space only added for clarity)
    """
    state = ""
    num_photons = sum(dist)
    qubits_per_mode = int(np.ceil(np.log2(num_photons+1)))
    for n in dist:
        bitstring = bin(n)[2:]
        bitstring = '0' * (qubits_per_mode - len(bitstring)) + bitstring # padding
        state += bitstring
    return state


def state_to_dist(state, num_modes):
    """
    Given a binary qubit state and the number of modes, returns the corresponding
    photon distribution

    Example:
    state = "001011", num_modes = 2
    >> [1,3] (One photon in first mode, 3 photons in second)
    """
    dist = []
    assert(len(state) % num_modes == 0)
    step_size = int(len(state) / num_modes)
    for i in range(0, len(state), step_size):
        substring = state[i:i+step_size]
        photons = int(substring, 2)
        dist.append(photons)
    return dist


def reverse_blocks(string, m):
    """
    This function splits a string up into blocks of length m, reverses the 
    order of the blocks and returns the resulting string
    """
    blocks = [string[i:i+m] for i in range(0, len(string), m)]
    reversed_blocks = blocks[::-1]
    reversed_string = ''.join(reversed_blocks)
    return reversed_string


def run_interferom_simulation(U, photon_config, num_shots):
    """
    Compile and run a quantum circuit corresponding to a linear interferometer with unitary U.

    U: Unitary matrix
    photon_config: The initial configuration (distribution) of photons.
        i.e. [5,7] means 5 photons in mode 0, 7 photon in mode 1.
    num_shots: The number of times the circuit is simulated
    """
    num_photons = sum(photon_config)
    num_modes = int(np.shape(U)[0])
    qubits_per_mode = int(np.ceil(np.log2(num_photons + 1)))
    num_qubits = qubits_per_mode * num_modes
    circuit = QuantumCircuit(num_qubits)
    
    initial_state = dist_to_state(photon_config)
    circuit.initialize(initial_state)
    interferom = direct_decomposition(U, num_photons)
    circuit.compose(interferom, qubits=list(range(num_qubits)), inplace=True)
    circuit.measure_all()

    simulator = AerSimulator()
    circuit = transpile(circuit, simulator) # TODO, this big circuit can be further optimized
    result = simulator.run(circuit, shots=num_shots).result()
    counts = result.get_counts(circuit)

    # Convert counts to probabilities
    for key in counts.keys():
        counts[key] = counts[key] / num_shots

    return counts
