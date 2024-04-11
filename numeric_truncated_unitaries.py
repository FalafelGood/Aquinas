"""
Methods for calculating numeric unitary matrices for beamsplitters 
that process up to k photons
"""

import numpy as np
from scipy.linalg import expm

def pad_matrix(M):
    """
    Given a square matrix M, this function returns a copy of M padded with zeros until 
    its shape is the next largest power of two.
    """

    def next_power_of_two(n):
        """
        Returns the next largest power of two greater than or equal to n.
        """
        return 2**int(np.ceil(np.log2(n)))
    
    n = M.shape[0]
    padded_size = next_power_of_two(n)
    padded_M = np.zeros((padded_size, padded_size), dtype=M.dtype)
    padded_M[:n, :n] = M
    return padded_M


def numeric_truncated_hamiltonian(theta, phi, k):
    """
    Calculates the numeric representation of a beamsplitter hamiltonian
    truncated at k photons for a given theta and phi.

    Recall that the Hamiltonian of a beamsplitter is given by
    $$ H = i e^{i \phi} \theta a_1^\dagger a_2 - i e^{-i \phi} \theta a_2^\dagger a_1 $$

    This function works by first calculating H explicitly for some fixed truncation depth n.
    """


    def a(n: int, mode: int):
        """
        Returns the annihilation operator for the 1st or 2nd beamsplitter mode truncated at n photons.
        
        If mode == 1, the operator is of the form:
            $$a_1 \otimes I_2$$

        Else if mode == 2, the operator is of the form 
            $$I_1 \otimes a_2$$

        Where $I_1$ and $I_2$ are the identity operators for modes one and two respectively

        I_1, I_2, a_1, a_2 are identically shaped (n+1 x n+1) operators.
        In the event where n + 1 is not a power of 2, then I_1, I_2, a_1, a_2
        will be padded with zeros until they are shaped to the next largest power of two.
        """
        assert(mode in (1,2)), "Invalid mode"
        d = n + 1 # The unpadded creation operator truncated at n photons has dimension dxd
        id = np.eye(d, d)
        matrix = np.zeros((d, d))
        for i in range(1, d):
            matrix[i-1, i] = np.sqrt(i)
        # Pad each component with zeros to the nearest power of two
        id = pad_matrix(id)
        matrix = pad_matrix(matrix)
        if mode == 1:
            return np.kron(matrix, id)
        return np.kron(id, matrix)
    

    def a_dag(n: int, mode: int):
        """
        Truncated creation operator
        """
        return np.transpose(a(n, mode))
    

    H = -1j * np.exp(1j*phi) * theta * a_dag(k, 1) @ a(k, 2) + 1j * np.exp(-1j*phi) * theta * a_dag(k, 2) @ a(k, 1)
    return H


def numeric_truncated_unitary(theta, phi, k):
    """
    Calculates the numeric representation of a beamsplitter unitary
    truncated at k photons for a given theta and phi.
    """
    H = numeric_truncated_hamiltonian(theta, phi, k)
    return expm(1j * H)
