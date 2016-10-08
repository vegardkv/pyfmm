import matplotlib.pyplot as plt
import pyfmm


my_image = plt.imread('irregular_boundary.png')
solution = pyfmm.march(my_image[:,:,0] == 0, batch_size=10)[0]

plt.imshow(solution, interpolation='None')
plt.colorbar()
plt.title('Irregular boundary')
plt.show()
