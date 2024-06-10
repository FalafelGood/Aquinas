"""
This is code borrowed from the Boson Sampling Library (Copyright (C) 2021  If and Only If (Iff) Technologies)
which falls under the GNU GENERAL PUBLIC LICENSE Version 3.

Repository:
https://github.com/IffTech/Boson-Sampling/tree/main

For some reason, this package isn't building for me. Jannis thought it was coderot in the .toml file.
I'm copying what I need since I don't have time to diagnose the problem and I only really need one function.

-Hudson
"""


import numpy as np
import scipy as sp
import thewalrus
from sympy.utilities.iterables import multiset_permutations
from strawberryfields.utils import random_interferometer


def gen_submatrix(photons_in, photons_out, unitary_mat):
    """Generates a submatrix from a given unitary matrix representing
    a linear interferometer whose permanent can be used to calculate
    photon output configuration probabilities.

    Args:
        photons_in ([int]):
            A list with each integer entry representing the number
            of individual photons in the input modes for the
            interferometer
        photons_out ([int]):
            A list with each integer entry representing the number
            of individual photons at the ouptut modes for the
            interferometer
        unitary_mat (np.array):
            A unitary matrix describing an interferometer

    Raises:
        ValueError:
            If the number of photons inputted do not equal the number
            of photons output, a ValueError is raised.

    Returns:
        np.array:
            A submatrix built from the originally inputted `unitary_mat`
            argument
    """

    if sum(photons_in) != sum(photons_out):
        raise ValueError("Number of photons inputted is not equal"
                         "to number outputted!")

    photon_count = sum(photons_in)

    # Generate a matrix consisting solely of columns from the main
    # unitary matrix. Row dimension is preserved but the number of
    # columns is equal to the number of photons.
    col_mat = np.zeros((unitary_mat.shape[0], photon_count),
                       dtype=np.cdouble)

    # Obtain columns for `col_mat`, based on the input photon
    # configuration.
    # Given a photon input [1,2,0,0], the first column of the main
    # unitary matrix is copied, then the second column is copied
    # TWICE, with no other columns copied to created an (N x 3) submatrix
    # (where N is whatever the row dimension is on the main unitary)
    col_idx = 0
    for col, val in enumerate(photons_in):
        for i in range(val):
            col_mat[:, col_idx] = unitary_mat[:, col]
            col_idx += 1

    # Create the final submatrix, which should be (number of photons in
    # x number of photons out) large
    sub_mat = np.zeros((photon_count, photon_count), dtype=np.cdouble)

    # Looking at the output photon configuration, rows from `col_mat`
    # are indexed and added to the final submatrix.
    # If the output is [1,2,0,0] then the first row of `col_mat` is
    # copied once, then the second row copied TWICE to create the final
    # submatrix (going off the previous example, should be 3 x 3)
    row_idx = 0
    for row, val in enumerate(photons_out):
        for i in range(val):
            sub_mat[row_idx, :] = col_mat[row, :]
            row_idx += 1

    return sub_mat


def output_probability(photons_in, photons_out, unitary_mat):
    """Calculate the probability of a certain photon output
    configuration (n number of photons across m modes) given the
    inputted photons and the unitary matrix representing a
    linear interferometer

    Args:
        photons_in ([int]):
            A list with each integer entry representing the number
            of individual photons in the input modes for the
            interferometer
        photons_out ([int]):
            A list with each integer entry representing the number
            of individual photons at the ouptut modes for the
            interferometer
        unitary_mat (np.array):
            A unitary matrix describing an interferometer

    Returns:
        float: probability photons will be detected in the given
        `photons_out` configuration
    """

    sub_mat = gen_submatrix(photons_in, photons_out, unitary_mat)

    modulus_squared = np.abs(thewalrus.perm(sub_mat))**2
    denom = (sp.special.factorial(photons_in).prod() *
             sp.special.factorial(photons_out).prod())

    return modulus_squared / denom

# """
# Here ends the code from the Boson sampling library
# """