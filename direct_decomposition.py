import interferometer as itf
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import U1Gate
from numeric_truncated_unitaries import *


def check_if_power_of_two(n):
    while True:
        if n == 1:
            return True
        if n % 2 == 0:
            n = n/2
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
    This wrapper function dictates how a given unitary matrix is decomposed into a quantum circuit.
    """
    dim = U.shape[0]
    assert check_if_power_of_two(dim)
    num_qubits = int(np.log2(dim))
    qc = QuantumCircuit(num_qubits)
    qc.unitary(U, range(num_qubits))
    compiled_qc = transpile(qc, basis_gates=['cx', 'u3'], optimization_level=2)

    # Compensate for global phase incurred in transpilation by rotating one qubit
    global_phase = compiled_qc.global_phase
    compensation_shift = U1Gate(-global_phase)
    compiled_qc.append(compensation_shift, [0])
    compiled_qc.global_phase = 0 # Manually set global phase to zero
    return compiled_qc


# Normal indexing
# def knit_qiskit_circuits(m, BS_list, circuits):
#     """
#     Knit a collection of qiskit beamsplitter circuits into a single interferometer

#     m: dimension of the interfeometer (equivalently the total number of modes)
#     """
#     qubits_per_bs = circuits[0].num_qubits
#     assert qubits_per_bs % 2 == 0 
#     """
#     A note on the above line: qubits_per_bs must be even since the beamsplitter
#     has two modes and therefore has two identically sized qubit
#     registers. The above assertion is only a formality to make sure
#     the next line doesn't cause any mischief.
#     """
#     qubits_per_mode = int(qubits_per_bs / 2) # Two modes per beamsplitter
#     total_num_qubits = qubits_per_mode * m
#     I_circ = QuantumCircuit(total_num_qubits)
#     for idx, circ in enumerate(circuits):
#         assert(BS_list[idx].mode1 < BS_list[idx].mode2) # Sanity check to make sure mode 1 is lower
#         lower_BS_mode = BS_list[idx].mode1 - 1 # Subtract one so mode count begins at zero
#         starting_qubit = lower_BS_mode * qubits_per_mode
#         acting_qubits = list(range(starting_qubit, starting_qubit + qubits_per_bs))
#         I_circ.compose(circ, qubits=acting_qubits, inplace=True)
#         I_circ.barrier() # debug for visualisation
#     return I_circ


# Attempt 2: Mirrored circuit -- Failed
# def knit_qiskit_circuits(m, BS_list, circuits):
#     """
#     Knit a collection of qiskit beamsplitter circuits into a single interferometer

#     m: dimension of the interfeometer (equivalently the total number of modes)
#     """
#     qubits_per_bs = circuits[0].num_qubits
#     assert qubits_per_bs % 2 == 0 
#     """
#     A note on the above line: qubits_per_bs must be even since the beamsplitter
#     has two modes and therefore has two identically sized qubit
#     registers. The above assertion is only a formality to make sure
#     the next line doesn't cause any mischief.
#     """
#     qubits_per_mode = int(qubits_per_bs / 2) # Two modes per beamsplitter
#     total_num_qubits = qubits_per_mode * m
#     I_circ = QuantumCircuit(total_num_qubits)
#     for idx, circ in enumerate(circuits):
#         assert(BS_list[idx].mode1 < BS_list[idx].mode2) # Sanity check to make sure mode 1 is lower
#         lower_BS_mode = BS_list[idx].mode1 - 1 # Subtract one so mode count begins at zero
#         starting_qubit = lower_BS_mode * qubits_per_mode
#         acting_qubits = list(range(starting_qubit, starting_qubit + qubits_per_bs))
#         I_circ.compose(circ, qubits=acting_qubits, inplace=True)
#         I_circ.barrier() # debug for visualisation
#     I_circ = I_circ.reverse_bits() # This is here so endian is consistent with boson_sampling_probabilities
#     return I_circ


# Attempt 3: Flipped indexing -- Failed
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
    qubits_per_mode = int(qubits_per_bs / 2) # Two modes per beamsplitter
    total_num_qubits = qubits_per_mode * m
    I_circ = QuantumCircuit(total_num_qubits)
    for idx, circ in enumerate(circuits):
        assert(BS_list[idx].mode1 < BS_list[idx].mode2) # Sanity check to make sure mode 1 is lower

        lower_BS_mode = BS_list[idx].mode1 - 1 # Subtract one so mode count begins at zero
        bottom_qubit = lower_BS_mode * qubits_per_mode
        
        # Move beamsplitter to the opposite half of the circuit
        bottom_qubit = total_num_qubits - (bottom_qubit + qubits_per_bs)
        acting_qubits = list(range(bottom_qubit, bottom_qubit + qubits_per_bs))
        I_circ.compose(circ, qubits=acting_qubits, inplace=True)
        I_circ.barrier() # debug for visualisation
    return I_circ


def direct_decomposition(U, k):
    """
    U: m*m unitary matrix representing an m mode interferometer
    k: Maximum number of photons that are expected at any given time
    """
    m = U.shape[0]
    circuits = []
    I = itf.square_decomposition(U) # type(I) == Interferometer
    for BS in I.BS_list:
        U_BS = numeric_truncated_unitary(BS.theta, BS.phi, k)
        circuits.append(compile_unitary(U_BS)) # Compile individual unitaries into quantum circuits
    interferometer_circuit = knit_qiskit_circuits(m, I.BS_list, circuits)
    return interferometer_circuit


# def initial_state_circuit(U, k):
#     return

# # import quantum_decomp
# quantum_decomp method of Fedoriaka et. al. -- Probably less efficient than qiskit's decomp
# def compile_unitary(U):
#     """
#     Wrapper function for unitary decomposition to quantum circuit
#     """
#     return quantum_decomp.matrix_to_qiskit_circuit(U)

# """
# Alternative strategy: compile the unitary matrix according to the openql implementation
# given by Krol et. al. which has better performance
# """
# def Krol_compilation(U):
#     return


### MAIN
# HOM = 1/np.sqrt(2) * np.matrix([[1,1],[1,-1]])
# circuits, interferom = direct_decomposition(HOM, 2)
# print(interferom)

# num_photons = 2
# U_inf = random_unitary(3)
# circuits, interferom = direct_decomposition(U_inf, num_photons)
# print(interferom)

# print(type(U_inf))

# circuits, interferom = direct_decomposition(Type1Fusion, 2)
# print(interferom.depth())


