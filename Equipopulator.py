from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt

def quadrature(vector):
    return sqrt(sum([i**2 for i in vector]))

def get_radius_given_number(number_in_circle):
    # asymptotic limit of number of circles that can be hexagonally packed in a square.
    equivalent_number_in_square = 4/pi * number_in_circle
    return 2/sqrt(equivalent_number_in_square*2*sqrt(3))

def tessellate_circle(num_sample):
    #create hexagonal packing of circles:
    r = get_radius_given_number(num_sample)
    xspace = 2*r
    yspace = 2*sqrt(3)*r
    full_rows = ary(np.meshgrid(np.arange(-1+r, 1, xspace), np.arange(-1+r, 1, yspace))).T.reshape([-1,2])
    in_between_rows = ary(np.meshgrid(np.arange(-1+2*r, 1, xspace), np.arange(-1+(1+sqrt(3))*r, 1, yspace))).T.reshape([-1,2])
    tessellated_square = np.concatenate([full_rows, in_between_rows])
    mask = [ quadrature(point)<1-0.8*r for point in tessellated_square]
    return tessellated_square[mask]

list_of_coordinates = tessellate_circle(450)
print(len(list_of_coordinates))
plt.scatter(*list_of_coordinates.T)
plt.show()