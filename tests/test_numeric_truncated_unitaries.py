# AMDG
import unittest
import numpy as np
from scipy.linalg import expm
from scipy.linalg import logm
from interferometer_circuits import numeric_truncated_unitaries


class TestBasicFunctions(unittest.TestCase):

    def test_random_unitaries(self):
        # Check that random_unitary and is_unitary methods are working
        self.assertTrue(
            numeric_truncated_unitaries.is_unitary(
                numeric_truncated_unitaries.random_unitary(5)
                )
            )
    
    def test_next_power_of_two(self):
        # Generic case
        assert(numeric_truncated_unitaries.next_power_of_two(195) == 256)
        # Edge case 1: n is already a power of two
        assert(numeric_truncated_unitaries.next_power_of_two(4) == 4)
        # Edge case 2:
        assert(numeric_truncated_unitaries.next_power_of_two(3) == 4)


    def test_pad_matrix(self):
        pow2 = np.array([[1,1],[2,2]])
        notpow2 = np.array([[1,1,1],[2,2,2],[3,3,3]])
        padded_pow2 = numeric_truncated_unitaries.pad_matrix(pow2)
        padded_notpow2 = numeric_truncated_unitaries.pad_matrix(notpow2)
        assert(np.array_equal(padded_pow2, pow2))

        indended_padding = np.array([[1,1,1,0],[2,2,2,0],[3,3,3,0],[0,0,0,0]])
        assert(np.array_equal(padded_notpow2, indended_padding))


class TestOperatorFunctions(unittest.TestCase):
    """
    Tests for the methods that create the (sometimes padded) annihilation operators
    No need to test creation operators since they're just transposed from annihilation.
    """

    def test_1mode_annihilation(self):
        # Test case zero: 1 photon
        a = numeric_truncated_unitaries.a
        op = a(n = 1, mode = 0, total_modes = 1)
        expected_op = np.array([[0,1],[0,0]])
        assert(np.array_equal(op, expected_op))

        # Test case 1: 3 photons. We expect no padding since 3 + 1 is a power of 2.
        big_op = a(n = 3, mode = 0, total_modes = 1)
        expected_big_op = np.array([[0,1,0,0],[0,0,np.sqrt(2),0],[0,0,0,np.sqrt(3)],[0,0,0,0]])
        assert(np.array_equal(big_op, expected_big_op))

        # Test case 2: 4 photons
        huge_op = a(n = 4, mode = 0, total_modes = 1)
        expected_huge_op = np.array([[0,1,0,0,0],
                                     [0,0,np.sqrt(2),0,0],
                                     [0,0,0,np.sqrt(3),0],
                                     [0,0,0,0,2],
                                     [0,0,0,0,0]])
        expected_huge_op = numeric_truncated_unitaries.pad_matrix(expected_huge_op)
        assert(np.array_equal(huge_op, expected_huge_op))


    def test_multimode_annihilation(self):
        a = numeric_truncated_unitaries.a

        # Test case one:
        op = a(n = 1, mode = 0, total_modes = 2)
        expected_op = a(n = 1 , mode = 0, total_modes = 1)
        expected_op = np.kron(expected_op, np.eye(2))
        assert(np.array_equal(op, expected_op))

        # Test case two: 
        op2 = a(n = 2, mode = 1, total_modes = 2)
        expected_op2 = a(n = 2, mode = 0, total_modes = 1)
        # Be careful! Identities are bigger and also padded
        padded_eye = numeric_truncated_unitaries.pad_matrix(np.eye(3))
        expected_op2 = np.kron(padded_eye, expected_op2)
        assert(np.array_equal(op2, expected_op2))

        # Test case three:
        op3 = a(n = 2, mode = 1, total_modes = 3)
        expected_op3 = a(n = 2, mode = 0, total_modes = 1)
        padded_eye = numeric_truncated_unitaries.pad_matrix(np.eye(3))
        expected_op3 = np.kron(padded_eye, np.kron(expected_op3, padded_eye))
        assert(np.array_equal(op3, expected_op3))


class TestNumericTruncatedUnitary(unittest.TestCase):
    """
    Tests for the numeric_truncated_unitary method in a variety of different scenarios that I was previously
    having trouble with myself.
    """

    def test_onephoton_beamsplitter(self):
        # Shorthand methods of creation and annihilation operators for improved readability
        def a(mode): 
            return numeric_truncated_unitaries.a(n = 1, mode = mode, total_modes = 2)

        def a_dag(mode):
            return numeric_truncated_unitaries.a_dag(n = 1, mode = mode, total_modes = 2)

        theta = 0.1
        phi = 0.1
        U_BS = np.array([[np.exp(1j * phi) * np.cos(theta), -np.sin(theta)],
                        [np.exp(1j * phi) * np.sin(theta), np.cos(theta)]])
        eye = np.eye(2)

        H_BS = -1j * logm(U_BS)

        truncated_H = H_BS[0,0] * a_dag(0) @ a(0) + H_BS[0,1] * a_dag(0) @ a(1) + \
        H_BS[1,0] * a_dag(1) @ a(0) + H_BS[1,1] * a_dag(1) @ a(1)

        # print(truncated_H)
        expected_unitary = expm(1j * truncated_H)
        U_trunc = numeric_truncated_unitaries.numeric_truncated_unitary(U = U_BS, n = 1, reverse_qubit_order = False)
        assert(np.array_equal(U_trunc, expected_unitary))


    def test_multi_photon_beamsplitter(self):
        num_photons = 3
        # Shorthand methods of creation and annihilation operators for improved readability
        # Same as before, now with three photons instead of one.
        def a(mode): 
            return numeric_truncated_unitaries.a(n = num_photons, mode = mode, total_modes = 2)

        def a_dag(mode):
            return numeric_truncated_unitaries.a_dag(n = num_photons, mode = mode, total_modes = 2)

        theta = 0.1
        phi = 0.1
        U_BS = np.array([[np.exp(1j * phi) * np.cos(theta), -np.sin(theta)],
                        [np.exp(1j * phi) * np.sin(theta), np.cos(theta)]])
        
        H_BS = -1j * logm(U_BS)

        truncated_H = H_BS[0,0] * a_dag(0) @ a(0) + H_BS[0,1] * a_dag(0) @ a(1) + \
        H_BS[1,0] * a_dag(1) @ a(0) + H_BS[1,1] * a_dag(1) @ a(1)

        # print(truncated_H)
        expected_unitary = expm(1j * truncated_H)
        U_trunc = numeric_truncated_unitaries.numeric_truncated_unitary(U = U_BS, n = num_photons, reverse_qubit_order = False)
        assert(np.allclose(U_trunc, expected_unitary))


    def test_beamsplitter_with_ancilla_mode(self):

        num_photons = 3

        def a(mode): 
            return numeric_truncated_unitaries.a(n = num_photons, mode = mode, total_modes = 3)

        def a_dag(mode):
            return numeric_truncated_unitaries.a_dag(n = num_photons, mode = mode, total_modes = 3)
        
        theta = 0.1
        phi = 0.1
        U_BS = np.array([[np.exp(1j * phi) * np.cos(theta), -np.sin(theta)],
                        [np.exp(1j * phi) * np.sin(theta), np.cos(theta)]])
        
        # U_itf is a unitary corresponding to an interferometer with a single beamsplitter
        # along the top two modes. The third mode does nothing
        U_itf = np.zeros((3,3), complex)
        U_itf[:2, :2] = U_BS
        U_itf[2, 2] = 1

        H_itf = -1j * logm(U_itf)

        truncated_H = H_itf[0,0] * a_dag(0) @ a(0) + H_itf[0,1] * a_dag(0) @ a(1) + \
        H_itf[1,0] * a_dag(1) @ a(0) + H_itf[1,1] * a_dag(1) @ a(1) \
        # All other H_itf elements are zero, there's no need to add other basis elements

        expected_unitary = expm(1j * truncated_H)
        U_trunc = numeric_truncated_unitaries.numeric_truncated_unitary(U = U_itf, n = num_photons, reverse_qubit_order = False)
        assert(np.allclose(U_trunc, expected_unitary))


    def test_arbitrary_3x3(self):
        """
        Test that numeric_truncated_unitary produces the correct unitary for an arbitrary 3x3 interferometer
        for a non-trivial number of photons
        """

        num_photons = 3

        def a(mode): 
            return numeric_truncated_unitaries.a(n = num_photons, mode = mode, total_modes = 3)

        def a_dag(mode):
            return numeric_truncated_unitaries.a_dag(n = num_photons, mode = mode, total_modes = 3)

        # An arbitrary 3x3 unitary
        U = [[0.808013 + 0.556201j, -0.180181 - 0.0664769j, 0.0295028 + 0.j], 
             [0.172644 + 0.0841329j, 0.918215 + 0.183126j, -0.294044 + 0.j], 
             [0.0281851 + 0.00871867j, 0.288183 + 0.0584175j, 0.955336 + 0.j]]
        
        H = -1j * logm(U)

        truncated_H = H[0,0] * a_dag(0) @ a(0) + H[0,1] * a_dag(0) @ a(1) + H[0,2] * a_dag(0) @ a(2) \
        + H[1,0] * a_dag(1) @ a(0) + H[1,1] * a_dag(1) @ a(1) + H[1,2] * a_dag(1) @ a(2) \
        + H[2,0] * a_dag(2) @ a(0) + H[2,1] * a_dag(2) @ a(1) + H[2,2] * a_dag(2) @ a(2)

        expected_unitary = expm(1j * truncated_H)
        U_trunc = numeric_truncated_unitaries.numeric_truncated_unitary(U = U, n = num_photons, reverse_qubit_order = False)
        assert(np.allclose(U_trunc, expected_unitary))


if __name__ == '__main__':
    unittest.main()