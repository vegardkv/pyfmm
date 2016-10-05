import pyfmm
import unittest
import numpy as np

DO_PLOT = False
if DO_PLOT:
    import matplotlib.pyplot as plt


class ApproximateDistanceTester(unittest.TestCase):
    def test_aprx_distances(self):
        my_mesh = np.zeros((5, 5), dtype=np.bool)
        certain = np.zeros((5, 5), dtype=np.bool)
        certain[2, 2] = True
        my_mesh[2, 2] = True
        uu = np.ones(my_mesh.shape) * np.inf
        uu[my_mesh] = 0
        new_vals = pyfmm.solvers.approximate_distance(uu, np.ones(my_mesh.shape), certain)

        solution = np.ones((5,5)) * np.inf
        solution[2, 2] = 0
        solution[1, 2] = 1
        solution[3, 2] = 1
        solution[2, 1] = 1
        solution[2, 3] = 1
        for i in range(1,5-1):
            for j in range(1,5-1):
                self.assertAlmostEqual(solution[i,j], new_vals[i,j])

    def test_aprx_distances_rectangular(self):
        my_mesh = np.zeros((5, 7), dtype=np.bool)
        certain = np.zeros((5, 7), dtype=np.bool)
        certain[2, 3] = True
        my_mesh[2, 3] = True
        uu = np.ones(my_mesh.shape) * np.inf
        uu[my_mesh] = 0
        new_vals = pyfmm.solvers.approximate_distance(uu, np.ones(my_mesh.shape), certain)

        solution = np.ones((5,7)) * np.inf
        solution[2, 3] = 0
        solution[1, 3] = 1
        solution[3, 3] = 1
        solution[2, 2] = 1
        solution[2, 4] = 1
        for i in range(1,5-1):
            for j in range(1,7-1):
                self.assertAlmostEqual(solution[i,j], new_vals[i,j])

    def test_aprx_circle(self):
        n = 300
        my_mesh = np.zeros((n, n), dtype=np.bool)
        certain = np.zeros((n, n), dtype=np.bool)
        certain[n/2, n/2] = True
        my_mesh[n/2, n/2] = True
        uu = np.ones(my_mesh.shape) * np.inf
        uu[my_mesh] = 0
        counter = 0
        while np.any(np.isinf(uu)) and counter < 1000:
            new_vals = pyfmm.solvers.approximate_distance(uu, np.ones(my_mesh.shape), certain)
            certain = True - np.isinf(new_vals)
            uu = new_vals
            counter += 1
        global DO_PLOT
        if DO_PLOT:
            plt.imshow(uu)
            plt.colorbar()
            plt.show()
        self.assertAlmostEqual(uu[1,1], 211, -1)
