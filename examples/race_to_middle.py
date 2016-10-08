import numpy as np
import matplotlib.pyplot as plt
import pyfmm

# Double speed, 50/50

my_mesh = np.zeros((20, 200), dtype=np.bool)
my_mesh[:,0] = True
my_mesh[:,-1] = True
speed_map = np.ones((20, 200))
speed_map[:, 100:] = 2.0

solution_a = pyfmm.march(my_mesh, speed=speed_map, batch_size=5)[0]
solution_b = pyfmm.march(my_mesh, speed=speed_map, batch_size=np.inf)[0]

plt.subplot(3,2,1)
plt.imshow(speed_map, interpolation='None')
plt.colorbar(orientation='horizontal')
plt.title('Speed map (50/50)')

plt.subplot(3,2,3)
plt.imshow(solution_a, interpolation='None')
plt.colorbar(orientation='horizontal')
plt.title('Accurate solution (50/50)')

plt.subplot(3,2,5)
plt.imshow(solution_b, interpolation='None')
plt.colorbar(orientation='horizontal')
plt.title('Inaccurate solution (50/50)')

# 10x speed, 90/10

my_mesh2 = np.zeros((20, 200), dtype=np.bool)
my_mesh2[:,0] = True
my_mesh2[:,-1] = True
speed_map2 = np.ones((20, 200))
speed_map2[:, 20:] = 10.0

solution_a2 = pyfmm.march(my_mesh2, speed=speed_map2, batch_size=5)[0]
solution_b2 = pyfmm.march(my_mesh2, speed=speed_map2, batch_size=100)[0]

plt.subplot(3,2,2)
plt.imshow(speed_map2, interpolation='None')
plt.colorbar(orientation='horizontal')
plt.title('Speed map (90/10)')

plt.subplot(3,2,4)
plt.imshow(solution_a2, interpolation='None')
plt.colorbar(orientation='horizontal')
plt.title('Accurate solution (90/10)')

plt.subplot(3,2,6)
plt.imshow(solution_b2, interpolation='None')
plt.colorbar(orientation='horizontal')
plt.title('Inaccurate solution (90/10)')

plt.show()

#TODO: should produce an example when batch_size = 40 produces a bad result (90% of left side is extremely fast, 10% of
# right side is slow. Plot this in a 3 x 2 subplot.
