# AMDG
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator
from interferometer_circuits.boson_sampling_probabilities import output_probability
from interferometer_circuits.direct_decomposition import direct_decomposition


def dist_to_state(dist, num_photons = None):
    """
    Given a photon distribution, returns the corresponding state as a binary string

    Example:
    dist := [1,3] (One photon in first mode, 3 photons in second)
    >> '001 011'
    (Note, space only added for clarity)

    If num_photons = None, the number of photons will be taken as the sum of 
    the elements of dist. Else, this method will return a state in a statespace
    that can support that number of photons.
    """
    state = ""
    if num_photons == None: 
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


def statevector_from_config(photon_config, little_endian = True, num_photons=None):
    """
    Given a photon configuration, this function returns the corresponding
    statevector as an numpy array
    """
    binrep = dist_to_state(photon_config, num_photons = num_photons) # binary representation of state
    if little_endian == True:
        binrep = binrep[::-1]
    
    num_qubits = len(binrep) # number of qubits in circuit
    statevector = np.zeros((2 ** num_qubits, 1), dtype=complex)
    nonzero_index = int(binrep, 2)
    statevector[nonzero_index] = 1.

    return statevector


def reverse_blocks(string, block_size):
    """
    This function splits a string up into blocks of length m, reverses the 
    order of the blocks and returns the resulting string
    """
    blocks = [string[i:i+block_size] for i in range(0, len(string), block_size)]
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
    
    initial_state = dist_to_state(photon_config)[::-1] # Reversed for qiskit's little endian
    circuit.initialize(initial_state)
    interferom = direct_decomposition(U, num_photons)
    circuit.compose(interferom, qubits=list(range(num_qubits)), inplace=True)
    circuit.measure_all()

    print(f"Num qubits {circuit.num_qubits}")
    print(f"Circuit depth = {circuit.depth()}")

    simulator = AerSimulator()
    circuit = transpile(circuit, simulator)
    result = simulator.run(circuit, shots=num_shots).result()
    counts = result.get_counts(circuit)

    # Convert counts to probabilities
    probs = dict()
    for key in counts.keys():
        # key is flipped since we no longer need to represent state in little endian
        output_dist = tuple(state_to_dist(key[::-1], num_modes = num_modes))
        probs[str(output_dist)] = counts[key] / num_shots

    return probs


def circuit_sampling_probability(input_config, output_config, interferometer_circuit):
    """
    Given an interferometer circuit, this function calculates the probability
    of measuring a particular output distribution of photons given some input
    distribution
    """
    circuit_U = Operator(interferometer_circuit).data
    ket = statevector_from_config(input_config, little_endian = True)
    bra = statevector_from_config(output_config, little_endian = True)
    amplitude = (np.transpose(bra) @ circuit_U @ ket)[0,0]
    return np.absolute(amplitude) ** 2