"""
Methods for calculating symbolic unitary matrices for beamsplitters 
of up to k photons
"""

import sympy as sp
import os
from sympy import exp, sqrt, I
from sympy.matrices import eye, zeros
from sympy.physics.quantum.tensorproduct import TensorProduct
import pickle

FOLDER = "symbolic_truncated_unitaries"


def check_unitary_exists(k):
    """
    Check that the symbolic unitary for a k photon beamsplitter 
    exists in /truncated_unitaries
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_name = f"{FOLDER}/{k}.pkl"
    file_path = os.path.join(script_directory, file_name)
    if os.path.exists(file_path):
        return True
    else:
        return False


def symbolic_truncated_unitary(k):
    """
    Symbolic unitary matrix for a beamsplitter of up to k photons
    """

    def a(n: int, mode: int):
        """
        Annihilation operator for the 1st or 2nd beamsplitter mode
        truncated at n photons.
        """
        assert(mode in (1,2)), "Invalid mode"
        d = n + 1 # The creation operator truncated at n photons has dimension dxd
        id = eye(d, d)
        matrix = zeros(d, d)
        for i in range(1, d):
            matrix[i-1, i] = sqrt(i)
        if mode == 1:
            return TensorProduct(matrix, id)
        return TensorProduct(id, matrix)


    def a_dag(n: int, mode: int):
        """
        Truncated creation operator for the 1st or 2nd beamsplitter mode
        truncated at n photons.
        """
        return a(n, mode).transpose()
    
    if check_unitary_exists(k): print(f"{k} photon unitary already exists"); return
    theta = sp.symbols('theta')
    phi = sp.symbols('phi')
    print(f"Calculating {k} photon unitary")
    H = I * exp(I*phi) * theta * a_dag(k, 1) * a(k, 2) - I * exp(-I*phi) * theta * a_dag(k, 2) * a(k, 1)
    U = sp.exp(I * H) # Don't bother simplifying for time being
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_name = f"truncated_unitaries/{k}.pkl"
    file_path = os.path.join(script_directory, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(U, f)
    return U


def fetch_symbolic_unitary(k):
    """
    Get k photon beamsplitter unitary from memory
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_name = f"{FOLDER}/{k}.pkl"
    file_path = os.path.join(script_directory, file_name)
    with open(file_path, 'rb') as f:
        U = pickle.load(f)
    return U


def sub_symbolic_unitary(U, theta_val, phi_val):
    """
    Substitute theta and phi values into a symbolic unitary
    """
    theta, phi = sp.symbols("theta, phi")
    return U.evalf(subs={theta:theta_val, phi:phi_val})

# Main
# U = fetch_symbolic_unitary(3)
# print(U.shape)