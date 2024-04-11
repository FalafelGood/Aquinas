import interferometer as itf
import numpy as np
from qiskit import QuantumCircuit, transpile
from numeric_truncated_unitaries import *


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


def round_complex_matrix(matrix, epsilon=1e-10):
    """
    Given a complex matrix (numpy.ndarray), round epsilon small numbers to zero 
    and 1-epsilon close numbers up to one.
    """
    small_indices = np.where(np.abs(matrix) < epsilon)
    close_to_one_indices = np.where(np.abs(matrix - 1) < epsilon)
    matrix[small_indices] = 0
    matrix[close_to_one_indices] = 1
    return matrix

def compile_unitary(U):
    """
    Wrapper function for unitary decomposition to quantum circuit
    """
    dim = U.shape[0]
    num_qubits = int(np.log2(dim))
    qc = QuantumCircuit(num_qubits)
    qc.unitary(U, range(num_qubits))
    compiled_qc = transpile(qc, basis_gates=['cx', 'u3'])
    return compiled_qc


def knit_qiskit_circuits(m, BS_list, circuits):
    """
    Knit a collection of qiskit beamsplitter circuits into a single interferometer

    m: dimension of the interfeometer (equivalently the total number of modes)
    """
    qubits_per_bs = circuits[0].num_qubits
    assert qubits_per_bs % 2 == 0 
    """
    Note: qubits_per_bs must be even since the beamsplitter
    has two modes and therefore two identically sized qubit
    registers. This assertion is only a formality to make sure
    the next line doesn't cause any mischief.
    """
    qubits_per_mode = int(qubits_per_bs / 2)
    total_num_qubits = qubits_per_mode * m # Remember m is equivalently the number of modes
    # total_num_qubits = int(np.ceil(m/2)) * qubits_per_bs # TODO Work to understand this.
    I_circ = QuantumCircuit(total_num_qubits)
    for idx, circ in enumerate(circuits):
        upper_BS_mode = BS_list[idx].mode1 - 1 # Subtract one to start the mode count at zero
        starting_qubit = upper_BS_mode * qubits_per_mode
        acting_qubits = list(range(starting_qubit, starting_qubit + qubits_per_bs))
        I_circ.compose(circ, qubits=acting_qubits, inplace=True)
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
        round_complex_matrix(U_BS)
        circuits.append(compile_unitary(U_BS)) # Break into circuits with Solovey-Kitaev
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


