# AMDG
import unittest
import numpy as np
import interferometer as itf
from interferometer_circuits import simulation
from interferometer_circuits import direct_decomposition
from interferometer_circuits import boson_sampling_probabilities


class ConversionMethodsTest(unittest.TestCase):

    def test_dist_to_state(self):
        photon_config = [1,0,0]
        assert(simulation.dist_to_state(photon_config) == "100")

        photon_config = [2,0,0]
        assert(simulation.dist_to_state(photon_config) == "100000")

        photon_config = [1,1,0]
        assert(simulation.dist_to_state(photon_config) == "010100")

        photon_config = [1,0,0]
        assert(simulation.dist_to_state(photon_config, num_photons=2) == "010000")

        photon_config = [2,1,1,1]
        # correct state: 010, 001, 001, 001
        assert(simulation.dist_to_state(photon_config) == "010001001001")


    def test_state_to_dist(self):
        state = "100"
        # == is fine for lists
        assert(simulation.state_to_dist(state, num_modes=3) == [1,0,0])

        state = "100000"
        assert(simulation.state_to_dist(state, num_modes=3) == [2,0,0])

        state = "010100"
        assert(simulation.state_to_dist(state, num_modes=3) == [1,1,0])

        state = "010001001001"
        assert(simulation.state_to_dist(state, num_modes=4) == [2,1,1,1])


    def test_statevector_from_config(self):

        photon_config = [1,0] # == |1> \otimes |0>
        sv = simulation.statevector_from_config(photon_config, little_endian=False)
        expected_sv = np.zeros((4,1), complex)
        expected_sv[2] = 1. 
        assert(np.array_equal(sv, expected_sv))

        # Test little_endian == True
        photon_config = [1,0] # == |0> \otimes |1> (little endian)
        sv = simulation.statevector_from_config(photon_config, little_endian=True)
        expected_sv = np.zeros((4,1), complex)
        expected_sv[1] = 1. 
        assert(np.array_equal(sv, expected_sv))

        # Test big statevector
        photon_config = [2,0,1] # == |10> \otimes |00> \otimes |01>
        nonzero_idx = 33 
        sv = simulation.statevector_from_config(photon_config, little_endian=False)
        expected_sv = np.zeros((64,1), complex)
        expected_sv[nonzero_idx] = 1.
        assert(np.array_equal(sv, expected_sv))


        # Test little endian on big statevector
        photon_config = [1,0,1] # == |10> \otimes |00> \otimes |10> (little endian)
        # In little
        nonzero_idx = 34
        sv = simulation.statevector_from_config(photon_config, little_endian=True)
        expected_sv = np.zeros((64,1), complex)
        expected_sv[nonzero_idx] = 1.
        assert(np.array_equal(sv, expected_sv))

        # Check that num_photons option works
        photon_config = [1,0]
        sv = simulation.statevector_from_config(photon_config, little_endian = False,
                                                num_photons=2)
        # Expected statevector: |01> \otimes |00> 
        nonzero_idx = 4
        expected_sv = np.zeros((16, 1), complex)
        expected_sv[nonzero_idx] = 1.
        assert(np.array_equal(sv, expected_sv))


    def test_reverse_blocks(self):
        initial_state = "100"
        expected_state = "001"
        state = simulation.reverse_blocks(initial_state, block_size=1)
        assert(expected_state == state)

        initial_state = "011011"
        expected_state = "111001"
        state = simulation.reverse_blocks(initial_state, block_size=2)
        assert(expected_state == state)


class CircuitSamplingTest(unittest.TestCase):
    def test_circuit_sampling_probability(self):
        BS_interferometer = itf.Interferometer()
        theta = 0.1
        phi = 0.1
        BS = itf.Beamsplitter(1, 2, theta=theta, phi=phi)
        BS_interferometer.add_BS(BS)
        U_BS = BS_interferometer.calculate_transformation()
        num_photons = 2
        BS_circuit = direct_decomposition.direct_decomposition(U_BS, num_photons)
        input_config = [2,0]
        output_config = [2,0]
        circuit_prob = simulation.circuit_sampling_probability(input_config, output_config, BS_circuit)
        expected_prob = boson_sampling_probabilities.output_probability(input_config, output_config, U_BS)
        assert(np.isclose(circuit_prob, expected_prob))


    def test_multi_mode_sampling(self):

        U = np.array([[0.808013 + 0.556201j, -0.180181 - 0.0664769j, 0.0295028 + 0.j], 
             [0.172644 + 0.0841329j, 0.918215 + 0.183126j, -0.294044 + 0.j], 
             [0.0281851 + 0.00871867j, 0.288183 + 0.0584175j, 0.955336 + 0.j]])
        
        num_photons = 3
        input_config = [2,1,0]
        output_config = [1,1,1]
        interferometer_circuit = direct_decomposition.direct_decomposition(U, num_photons)
        circuit_prob = simulation.circuit_sampling_probability(input_config, output_config, interferometer_circuit)
        expected_prob = boson_sampling_probabilities.output_probability(input_config, output_config, U)
        assert(circuit_prob == expected_prob)



if __name__ == '__main__':
    unittest.main()