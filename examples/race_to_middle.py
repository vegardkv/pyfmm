import numpy as np
import matplotlib.pyplot as plt
import pyfmm

my_mesh = np.zeros((20, 200), dtype=np.bool)
my_mesh[:,0] = True
my_mesh[:,-1] = True
speed_map = np.ones((20, 200))
speed_map[:, 100:] = 2.0

solution_a = pyfmm.march(my_mesh, speed=speed_map, batch_size=20)[0]
solution_b = pyfmm.march(my_mesh, speed=speed_map, batch_size=np.inf)[0]

plt.subplot(3,1,1)
plt.imshow(speed_map, interpolation='None')
plt.colorbar()
plt.title('Speed map')

plt.subplot(3,1,2)
plt.imshow(solution_a, interpolation='None')
plt.colorbar()
plt.title('Accurate solution')

plt.subplot(3,1,3)
plt.imshow(solution_b, interpolation='None')
plt.colorbar()
plt.title('Inaccurate solution')

plt.show()

#TODO: should produce an example when batch_size = 40 produces a bad result (90% of left side is extremely fast, 10% of
# right side is slow. Plot this in a 3 x 2 subplot.
