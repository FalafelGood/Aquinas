# AMDG
import interferometer as itf
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import U1Gate
from Aquinas.numeric_truncated_unitaries import *


def check_if_power_of_two(n):
    """
    Hello Alan! I will learn your bithacking ways another day.
    For now, enjoy your aneurysm!
    """
    while True:
        if n == 1:
            return True
        if n % 2 == 0:
            n = n / 2
        else:
            return False
        

def random_unitary(N):
    """Returns a random NxN unitary matrix

    Code credit: clementsw
    https://github.com/clementsw/interferometer/blob/master/interferometer/main.py
    """
    X = np.zeros([N, N], dtype=np.complex_)
    for ii in range(N):
        for jj in range(N):
            X[ii, jj] = (np.random.normal() + 1j * np.random.normal()) / np.sqrt(2)

    q, r = np.linalg.qr(X)
    r = np.diag(np.divide(np.diag(r), abs(np.diag(r))))
    U = np.matmul(q, r)
    return U


def compile_unitary(U):
    """
    This wrapper function takes a unitary matrix and transpiles it into a quantum circuit.
    Right now, we use qiskit's transpile method at optimization_level=2
    """
    dim = U.shape[0]
    assert check_if_power_of_two(dim)
    num_qubits = int(np.log2(dim))
    qc = QuantumCircuit(num_qubits)
    qc.unitary(U, range(num_qubits))
    compiled_qc = transpile(qc, basis_gates=['cx', 'u3'], optimization_level=2)
    return compiled_qc


# Normal indexing
def knit_qiskit_circuits(m, BS_list, circuits):
    """
    Knit a collection of qiskit beamsplitter circuits into a single interferometer

    m: dimension of the interfeometer (equivalently the total number of modes)
    """
    qubits_per_bs = circuits[0].num_qubits
    assert qubits_per_bs % 2 == 0 
    """
    A note on the above line: qubits_per_bs must be even since the beamsplitter
    has two modes and therefore has two identically sized qubit
    registers. The above assertion is only a formality to make sure
    the next line doesn't cause any mischief.
    """
    qubits_per_mode = qubits_per_bs // 2 # Two modes per beamsplitter
    total_num_qubits = qubits_per_mode * m
    I_circ = QuantumCircuit(total_num_qubits)
    for idx, circ in enumerate(circuits):
        assert(BS_list[idx].mode1 < BS_list[idx].mode2) # Sanity check to make sure mode 1 is lower
        lower_BS_mode = BS_list[idx].mode1 - 1 # Subtract one so mode count begins at zero
        starting_qubit = lower_BS_mode * qubits_per_mode
        acting_qubits = list(range(starting_qubit, starting_qubit + qubits_per_bs))
        I_circ.compose(circ, qubits=acting_qubits, inplace=True)
        I_circ.barrier() # debug for visualisation
    return I_circ


def BS_unitary(theta, phi):
    """
    Returns the unitary corresponding to a beamsplitter parameterised by 
    theta (related to reflectivity) and phi (relative phase)
    """
    return np.array([[np.exp(1j * phi) * np.cos(theta), -np.sin(theta)],
                     [np.exp(1j * phi) * np.sin(theta), np.cos(theta)]])


def direct_decomposition(U, k):
    """
    U: m * m unitary matrix representing an m mode interferometer
    k: Maximum number of photons that are expected at any given time
    """
    m = U.shape[0]
    circuits = []
    I = itf.square_decomposition(U) # type(I) == Interferometer
    for BS in I.BS_list:
        U_BS = BS_unitary(BS.theta, BS.phi)
        U_trunc = numeric_truncated_unitary(U_BS, k, reverse_qubit_order=True)
        circuits.append(compile_unitary(U_trunc)) # Compile individual unitaries into quantum circuits
    interferometer_circuit = knit_qiskit_circuits(m, I.BS_list, circuits)
    
    # TODO, is this code working / doing anything?
    # Add phases: 
    # qubits_per_mode = int(np.log2(next_power_of_two(k+1)))
    # for mode_idx, out_phi in enumerate(I.output_phases):
    #     starting_qubit = qubits_per_mode * mode_idx
    #     # finishing_qubit = qubits_per_mode * (mode_idx + 1)
    #     # acting_qubits = range(starting_qubit, finishing_qubit)
    #     # interferometer_circuit.rz(-out_phi, acting_qubits)
    #     interferometer_circuit.rz(-out_phi, starting_qubit)

    return interferometer_circuit


def decompose_from_interferom(I, k):
    """
    I: itf.Interferometer
    k: Maximum number of photons that are expected at any given time
    """
    m = I.count_modes()
    circuits = []
    for BS in I.BS_list:
        U_BS = BS_unitary(BS.theta, BS.phi)
        U_trunc = numeric_truncated_unitary(U_BS, k, reverse_qubit_order=True) 
        # Note: reverse_qargs = True is needed for Qiskit's little endian nonsense.
        circuits.append(compile_unitary(U_trunc)) # Compile individual unitaries into quantum circuits
    interferometer_circuit = knit_qiskit_circuits(m, I.BS_list, circuits)
    
    # Add phases:
    # qubits_per_mode = int(np.log2(next_power_of_two(k+1)))
    # for mode_idx, out_phi in enumerate(I.output_phases):
    #     starting_qubit = qubits_per_mode * mode_idx
    #     finishing_qubit = qubits_per_mode * (mode_idx + 1)
    #     acting_qubits = range(starting_qubit, finishing_qubit)
    #     interferometer_circuit.rz(-out_phi, acting_qubits)

    return interferometer_circuit