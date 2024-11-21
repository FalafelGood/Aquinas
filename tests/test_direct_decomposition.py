# AMDG
import unittest
from interferometer_circuits import direct_decomposition, numeric_truncated_unitaries
import interferometer as itf
import numpy as np
from qiskit.quantum_info import Operator


class IdentityBeamsplitterTest(unittest.TestCase):
    """
    Tests to see if a single idempodent beamsplitter transpiles to idempotent quantum
    circuits for one and more photons.
    """

    interferometer = itf.Interferometer()
    theta = 0.0
    phi = 0.0
    BS = itf.Beamsplitter(1, 2, theta=theta, phi=phi)
    interferometer.add_BS(BS)
    U_BS = interferometer.calculate_transformation()

    def test_beamsplitter_identity(self):
        """
        Make a quantum circuit that simulates a 'do-nothing' beamsplitter up to one photon.

        For one photon, one qubit is required per mode to encode the photon number
        
        e.g.
        |0> := no photon present in mode
        |1> := one photon present in mode
        """
        num_photons = 1
        total_num_qubits = 2 * 1 # 2 modes, 1 qubit per mode
        qc = direct_decomposition.direct_decomposition(IdentityBeamsplitterTest.U_BS , num_photons)
        U_circ = Operator(qc).data
        identity = np.eye(2 ** total_num_qubits)
        assert(np.allclose(U_circ, identity))


    def test_multiphoton_beamsplitter_identity(self):
        """
        Make a quantum circuit that simulates a 'do nothing' beamsplitter up to five photons.

        For five photons, three qubits are required to encode the photon number of each mode 
        since 5 == 101b
        """
        num_photons = 5
        total_num_qubits = 2 * 3 # 2 modes, 3 qubits per mode
        qc = direct_decomposition.direct_decomposition(IdentityBeamsplitterTest.U_BS , num_photons)
        U_circ = Operator(qc).data
        identity = np.eye(2 ** total_num_qubits)
        assert(np.allclose(U_circ, identity))


class IdentityInterferometerTest(unittest.TestCase):
    """
    Tests to see if a four mode idempotent interferometer transpiles to an idempotent
    circuit for one or more photons

    NOTE: Global phase starts to be introduced when multiple beamsplitters are "stitched"
    together. This test checks that the two unitaries are equivalent up to global phase.
    """

    interferometer = itf.Interferometer()
    theta = 0.0
    phi = 0.0
    BS1 = itf.Beamsplitter(1, 2, theta=theta, phi=phi)
    BS2 = itf.Beamsplitter(3, 4, theta=theta, phi=phi)
    interferometer.add_BS(BS1)
    interferometer.add_BS(BS2)
    U_itf = interferometer.calculate_transformation()

    def test_beamsplitter_identity(self):
        """
        Make a quantum circuit that simulates a 'do-nothing' four mode interferometer 
        up to one photon.

        As with the previous tests, one qubit is required per mode to encode the photon number
        """
        num_photons = 1
        total_num_qubits = 4 * 1 # 4 modes, 1 qubit per mode
        qc = direct_decomposition.direct_decomposition(IdentityInterferometerTest.U_itf , num_photons)
        U_circ = Operator(qc)
        identity = np.eye(2 ** total_num_qubits)
        U_circ.equiv(Operator(identity)) # Check if equivalent up to global phase


    def test_multiphoton_beamsplitter_identity(self):
        """
        Make a quantum circuit that simulates a 'do nothing' four mode interferometer
        up to three photons

        For three photons, two qubits are required to encode the photon number of each mode 
        since 3 == 11b
        """
        num_photons = 3
        total_num_qubits = 4 * 2 # 4 modes, 2 qubits per mode
        qc = direct_decomposition.direct_decomposition(IdentityBeamsplitterTest.U_BS , num_photons)
        U_circ = Operator(qc)
        identity = np.eye(2 ** total_num_qubits)
        U_circ.equiv(Operator(identity)) # Check if equivalent up to global phase


class NonTrivialBeamsplitterTest(unittest.TestCase):
    """
    Tests to see if a single beamsplitter transpiles to the correct quantum
    circuits for one and more photons.
    """

    interferometer = itf.Interferometer()
    theta = 0.3
    phi = 0.1
    BS = itf.Beamsplitter(1, 2, theta=theta, phi=phi)
    interferometer.add_BS(BS)
    U_BS = interferometer.calculate_transformation()


    def test_beamsplitter(self):
        """
        Make a quantum circuit that simulates an arbitrary beamsplitter up to one photon.
        """
        num_photons = 1
        total_num_qubits = 2 * 1 # 2 modes, 1 qubit per mode
        qc = direct_decomposition.direct_decomposition(NonTrivialBeamsplitterTest.U_BS , num_photons)
        U_circ = Operator(qc).data
        U_trunc = numeric_truncated_unitaries.numeric_truncated_unitary(NonTrivialBeamsplitterTest.U_BS, num_photons)
        assert(np.allclose(U_circ, U_trunc))


    def test_multiphoton_beamsplitter(self):
        """
        Make a quantum circuit that simulates an arbitrary beamsplitter up to five photons.
        """
        num_photons = 5
        total_num_qubits = 2 * 3 # 2 modes, 3 qubit per mode
        qc = direct_decomposition.direct_decomposition(NonTrivialBeamsplitterTest.U_BS , num_photons)
        U_circ = Operator(qc).data
        U_trunc = numeric_truncated_unitaries.numeric_truncated_unitary(NonTrivialBeamsplitterTest.U_BS, num_photons)
        assert(np.allclose(U_circ, U_trunc))


class NonTrivialInterferometerTest(unittest.TestCase):
    """
    Tests to see if a three mode interferometer transpiles to the correct quantum
    circuits for one and more photons.
    """

    interferometer = itf.Interferometer()
    theta_1 = 0.1
    phi_1 = 0.1
    
    theta_2 = 0.3
    phi_2 = 0.2
    
    theta_3 = 0.1
    phi_3 = 0.5

    BS1 = itf.Beamsplitter(1, 2, theta=theta_1, phi=phi_1)
    BS2 = itf.Beamsplitter(2, 3, theta=theta_2, phi=phi_2)
    BS3 = itf.Beamsplitter(1, 2, theta=theta_3, phi=phi_3)

    interferometer.add_BS(BS1)
    interferometer.add_BS(BS2)
    interferometer.add_BS(BS3)

    U_itf = interferometer.calculate_transformation()

    def test_interferometer(self):
        """
        Test arbitrary 3 mode interferometer for one photon
        """
        num_photons = 1
        qc = direct_decomposition.direct_decomposition(NonTrivialInterferometerTest.U_itf, num_photons)
        U_circ = Operator(qc).data
        U_trunc = numeric_truncated_unitaries.numeric_truncated_unitary(NonTrivialInterferometerTest.U_itf, num_photons)
        
        # Statevectors for |001>, |010>, |100>
        state001 = np.array([0,1,0,0,0,0,0,0], complex)
        state010 = np.array([0,0,1,0,0,0,0,0], complex)
        state100 = np.array([0,0,0,0,1,0,0,0], complex)

        assert(np.allclose(U_circ @ state001, U_trunc @ state001))
        assert(np.allclose(U_circ @ state010, U_trunc @ state010))
        assert(np.allclose(U_circ @ state100, U_trunc @ state100))

if __name__ == '__main__':
    unittest.main()