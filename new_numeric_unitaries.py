import numpy as np
from scipy.linalg import expm
from scipy.linalg import logm
from copy import copy


# def random_unitary(N):
#     """Returns a random NxN unitary matrix

#     Code credit: clementsw
#     https://github.com/clementsw/interferometer/blob/master/interferometer/main.py
#     """
#     X = np.zeros([N, N], dtype=np.complex_)
#     for ii in range(N):
#         for jj in range(N):
#             X[ii, jj] = (np.random.normal() + 1j * np.random.normal()) / np.sqrt(2)

#     q, r = np.linalg.qr(X)
#     r = np.diag(np.divide(np.diag(r), abs(np.diag(r))))
#     U = np.matmul(q, r)
#     return U


# def is_unitary(m):
#     return np.allclose(np.eye(m.shape[0]), np.conjugate(np.transpose(m)) @ m)


def next_power_of_two(n):
    """
    Returns the next largest power of two greater than or equal to n.
    """
    return 2**int(np.ceil(np.log2(n)))


def pad_matrix(M):
    """
    Given a square matrix M, this function returns a copy of M padded with zeros until 
    its shape is the next largest power of two.
    """
    
    n = M.shape[0]
    padded_size = next_power_of_two(n)
    padded_M = np.zeros((padded_size, padded_size), dtype=M.dtype)
    padded_M[:n, :n] = M
    return padded_M


def a(n: int, mode: int, total_modes: int):
    """
    Returns the annihilation operator for the 1st or 2nd beamsplitter mode truncated at n photons.
    
    If mode == 1 and total_modes == 2, the operator is of the form:
        $$a_1 \otimes I_2$$

    Else if mode == 2, the operator is of the form 
        $$I_1 \otimes a_2$$

    Where $I_1$ and $I_2$ are the identity operators for modes one and two respectively

    Note: 
    I_1, I_2, a_1, a_2 are identically shaped (n+1 x n+1) operators.
    In the event where n + 1 is not a power of 2, then I_1, I_2, a_1, a_2
    will be padded with zeros until they are shaped to the next largest power of two.
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


def numeric_truncated_unitary(U, n):

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
    # i,j (row, column)
    for i, row in enumerate(logU):
        for j, element in enumerate(row):
            # The creation operator should act on column index (j)
            # The annihilation operator should act on row index (i)
            truncated_hamiltonian += element * a_dag(n, j, num_modes) @ a(n, i, num_modes)
    return expm(1j * truncated_hamiltonian)


# Test
import interferometer as itf
from random import random
I = itf.Interferometer()
theta = 0.333
phi = 0.667

BS = itf.Beamsplitter(1, 2, theta=theta, phi=phi)
I.add_BS(BS)

U = I.calculate_transformation()
truncated_U = numeric_truncated_unitary(U, 1)
print(truncated_U)

# print("\n")

# old_truncated_U = old_numeric_truncated_unitary(theta, phi, 1)
# print(old_truncated_U)

# U = numeric_truncated_unitary(random_unitary(2), 1)
# print(U)

# print(np.array_equal(a(5, 0, 2), old_a(5, 1)))
# print(np.shape((a(1,0,2))))
# print(old_a(1,1))

# print(size_of_truncated_unitary(1,2))

# def size_of_truncated_unitary(n, num_modes):
#     unpadded_op = n + 1 # size of unpadded single mode creation / identity operator
#     padded_op = next_power_of_two(unpadded_op) # size of padded operator
#     size = padded_op ** num_modes
#     return size

# Test on random unitaries