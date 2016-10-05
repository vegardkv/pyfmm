import pyfmm
import unittest
import numpy as np

DO_PLOT = False
if DO_PLOT:
    import matplotlib.pyplot as plt


class FMMSolverTester(unittest.TestCase):
    def setUp(self):
        super().setUp()
        global DO_PLOT
        self.do_plot = DO_PLOT

    def test_fmm(self):
        # Given
        my_mesh = np.zeros((5, 5), dtype=np.bool)
        my_mesh[2, 2] = True

        # When
        solution = pyfmm.march(my_mesh, batch_size=1)[0]

        # Then
        if self.do_plot:
            plt.imshow(solution, interpolation='None')
            plt.colorbar()
            plt.show()
        self.assertFalse(np.argwhere(np.isnan(solution)))
        self.assertAlmostEqual(solution[2,2], 0)
        self.assertAlmostEqual(solution[0,2], 2.0)
        self.assertAlmostEqual(solution[2,0], 2.0)

    def test_fmm_npick(self):
        # Given
        my_mesh = np.zeros((5, 5), dtype=np.bool)
        my_mesh[2, 2] = True

        # When
        solution = pyfmm.march(my_mesh, batch_size=5)[0]

        # Then
        if self.do_plot:
            plt.imshow(solution, interpolation='None')
            plt.colorbar()
            plt.show()
        self.assertFalse(np.any(np.isnan(solution)))
        self.assertAlmostEqual(solution[2, 2], 0)
        self.assertAlmostEqual(solution[0, 2], 2.0)
        self.assertAlmostEqual(solution[2, 0], 2.0)

    def test_fmm_infpick(self):
        # Given
        my_mesh = np.zeros((5, 5), dtype=np.bool)
        my_mesh[2, 2] = True

        # When
        exact = pyfmm.march(my_mesh, batch_size=np.inf)[0]

        # Then
        if self.do_plot:
            plt.imshow(exact, interpolation='None')
            plt.colorbar()
            plt.show()
        self.assertFalse(np.argwhere(np.isnan(exact)))
        self.assertAlmostEqual(exact[2, 2], 0)
        self.assertAlmostEqual(exact[0, 2], 2.0)
        self.assertAlmostEqual(exact[2, 0], 2.0)

    def test_fmm_L_shape(self):
        # Given
        my_mesh = np.zeros((200, 200), dtype=np.bool)
        my_mesh[75:125, 75] = True
        my_mesh[75, 75:125] = True

        # When
        solution = pyfmm.march(my_mesh, batch_size=100)[0]
        #solution = pyfmm.pyfmm(my_mesh, n_min_pick_size=20)  TODO: investigate why some values are undefined (inf or nan)

        # Then
        if self.do_plot:
            plt.imshow(solution, interpolation='None')
            plt.colorbar()
            plt.show()
        self.assertFalse(np.any(np.isinf(solution)))
        self.assertFalse(np.any(np.isnan(solution)))
        self.assertTrue(np.max(solution) < 160)

    def test_fmm_race_to_middle(self):
        # Given
        my_mesh = np.zeros((20, 200), dtype=np.bool)
        my_mesh[:,0] = True
        my_mesh[:,-1] = True
        speed_map = np.ones((20, 200))
        speed_map[:, 100:] = 2.0

        # When
        solution = pyfmm.march(my_mesh, speed=speed_map, batch_size=20)[0]

        # Then
        if self.do_plot:
            plt.imshow(solution, interpolation='None')
            plt.colorbar()
            plt.show()
        self.assertFalse(np.any(np.isinf(solution)))
        self.assertFalse(np.any(np.isnan(solution)))
        self.assertTrue(np.max(solution) < 75)

    def test_fmm_plateau(self):
        # Given
        my_mesh = np.zeros((20, 200), dtype=np.bool)
        my_mesh[:, 0] = True
        speed_map = np.ones((20, 200))
        speed_map[:, 100:] = 0.0

        # When
        solution, certain_values = pyfmm.march(my_mesh, speed=speed_map, batch_size=20)

        # Then
        if self.do_plot:
            plt.imshow(solution, interpolation='None')
            plt.colorbar()
            plt.show()
        self.assertLessEqual(np.sum(np.isinf(solution)), 2000)
        self.assertLessEqual(np.sum(certain_values), 2000)

    def test_fmm_circle(self):
        # Given
        n = 200
        my_mesh = np.zeros((n,n), dtype=np.bool)
        xx = np.array(100 + 50*np.cos(np.linspace(0, 2*np.pi, 100)), dtype=np.int)
        yy = np.array(100 + 50*np.sin(np.linspace(0, 2*np.pi, 100)), dtype=np.int)
        my_mesh[xx, yy] = True

        # When
        solution = pyfmm.march(my_mesh, batch_size=10)[0]

        # Then
        if self.do_plot:
            plt.imshow(solution, interpolation='None')
            plt.colorbar()
            plt.show()
        self.assertFalse(np.any(np.isinf(solution)))
        self.assertAlmostEqual(solution[n/2,n/2], 50.0, places=-1)
        self.assertAlmostEqual(solution[0, 0], 1.4142*100 - 50, places=-1)

    def test_fmm_known_values(self):
        # Given
        xx = np.outer(np.arange(0, 20, 1), np.ones((20,)))
        yy = np.outer(np.ones((20,)), np.arange(0, 20, 1))
        true_solution = np.abs(np.sqrt(np.square(xx) + np.square(yy)) - 10)
        known_values = true_solution < 1.0

        # When
        computed_solution, _discard = pyfmm.march(known_values, true_solution, batch_size=1)

        # Then
        if self.do_plot:
            plt.imshow(computed_solution - true_solution, interpolation='None')
            plt.colorbar()
            plt.title('test_fmm_known_values')
            plt.show()

        self.assertLess(np.max(np.abs(computed_solution - true_solution)), 0.7)
