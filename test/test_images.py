import pyfmm
import unittest
import numpy as np
import scipy.misc

DO_PLOT = True
if DO_PLOT:
    import matplotlib.pyplot as plt

class ImageTester(unittest.TestCase):
    def setUp(self):
        super().setUp()
        global DO_PLOT
        self.do_plot = DO_PLOT

    def test_sr_1(self):
        img = scipy.misc.imread('img/sr1.png', flatten=True)
        speed = (img > 0)*1.0
        my_mesh = np.zeros(speed.shape, dtype=np.bool)
        my_mesh[256, 256] = True
        solution = pyfmm.march(my_mesh, speed=speed, batch_size=50)

        if self.do_plot:
            plt.imshow(solution[0])
            plt.colorbar()
            plt.show()
        self.assertFalse(np.argwhere(np.isnan(solution)))

if __name__ == '__main__':
    pass