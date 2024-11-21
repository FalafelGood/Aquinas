# AMDG
import numpy as np
from scipy.linalg import expm
from scipy.linalg import logm
from copy import copy
from qiskit.quantum_info import Operator


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


def is_unitary(m):
    return np.allclose(np.eye(m.shape[0]), np.conjugate(np.transpose(m)) @ m)


def next_power_of_two(n):
    """
    Returns the next largest power of two.
    If n is a power of two, its power is returned
    """
    return 1 << int(np.ceil(np.log2(n)))


def pad_matrix(M):
    """
    Given a square matrix M, this function returns a copy of M padded with zeros until 
    its shape is the next largest power of two.

    If the shape of M is already a power of two, no padding occours.
    """
    
    n = M.shape[0]
    padded_size = next_power_of_two(n)
    padded_M = np.zeros((padded_size, padded_size), dtype=M.dtype)
    padded_M[:n, :n] = M
    return padded_M


def a(n: int, mode: int, total_modes: int):
    """
    Returns the annihilation operator for the 1st or 2nd beamsplitter mode truncated at n photons.
    
    If mode == 0 and total_modes == 2, the operator is of the form:
        $$a_0 \otimes I_1$$

    Else if mode == 1, the operator is of the form 
        $$I_0 \otimes a_1$$

    Where $I_0$ and $I_1$ are the identity operators for modes zero and one respectively

    Note: 
    I_0, I_1, a_0, a_1 are identically shaped (n+1 x n+1) operators.
    In the event where n + 1 is not a power of 2, then I_0, I_1, a_0, a_1
    will be padded with zeros until they are shaped to the next largest power of two.

    This is to ensure that numeric_truncated_unitary() returns a matrix that can be transpiled
    directly into a quantum circuit
    """

    assert(mode < total_modes), "Invalid mode"
    d = n + 1 # The unpadded creation operator truncated at n photons has dimension dxd
    id = np.eye(d, d)
    a = np.zeros((d, d))
    for i in range(1, d):
        a[i-1, i] = np.sqrt(i)
    # Pad each component with zeros to the nearest power of two
    id = pad_matrix(id)
    a = pad_matrix(a)

    if mode == 0:
        operator = copy(a)
    else:
        operator = copy(id)

    for i in range(1, total_modes):
        if i == mode:
            operator = np.kron(operator, a)
        else:
            operator = np.kron(operator, id)
    
    return operator


def a_dag(n: int, mode: int, total_modes: int):
    """
    Truncated creation operator
    """
    return np.transpose(a(n, mode, total_modes))


def reverse_qargs(U):
    """
    Takes a unitary U corresponding to a qubit circuit and returns a unitary
    equivalent to the same circuit but with a flipped qubit order.
    """
    return Operator(U).reverse_qargs().data


def numeric_truncated_unitary(U, n, reverse_qubit_order=True):
    """
    U: numpy array (unitary matrix) corresponding to a linear interferometer
    n: Maximum number of photons the truncated unitary can support
    reverse_qubit_order: If true, uses little endian qubit ordering
    """

    def size_of_truncated_unitary():
        num_modes = np.shape(U)[0]
        unpadded_op = n + 1 # size of unpadded single mode creation / identity operator
        padded_op = next_power_of_two(unpadded_op) # size of padded operator
        size = padded_op ** num_modes
        return size
    
    num_modes = np.shape(U)[0]
    size = size_of_truncated_unitary()
    truncated_hamiltonian = np.zeros((size, size), dtype=complex)
    logU = -1j * logm(U)
    # i,j (row idx, column idx)
    for i, row in enumerate(logU):
        for j, element in enumerate(row):
            # The creation operator should act on row index (j)
            # The annihilation operator should act on column index (i)
            truncated_hamiltonian += element * a_dag(n, i, num_modes) @ a(n, j, num_modes)
    U_trunc = expm(1j * truncated_hamiltonian)
    
    if reverse_qubit_order == True:
        return reverse_qargs(U_trunc)
    else:
        return U_trunc

