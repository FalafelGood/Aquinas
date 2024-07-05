# AMDG
import unittest
from interferometer_circuits import simulation
import numpy as np

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
        print(simulation.dist_to_state([1,0], num_photons=2))
        # Expected statevector: |01> \otimes |00> 
        nonzero_idx = 4
        expected_sv = np.zeros((16, 1), complex)
        expected_sv[nonzero_idx] = 1.
        assert(np.array_equal(sv, expected_sv))




if __name__ == '__main__':
    unittest.main()