from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
from SimpleMapping import radius_max

height_of_twist = 75.6 #mm
outer_radius_of_cable = 1.98 #mm
# inner_radius_of_cable = 0.45 #mm
INV_ASPECT_RATIO = height_of_twist/outer_radius_of_cable
HEIGHT_OF_TWIST = radius_max * INV_ASPECT_RATIO

def rotation_mat(theta):
    """
    returns a rotation matrix that rotates anti-clockwise by theta radians
    """
    return ary([
    [cos(theta), -sin(theta)],
    [sin(theta),  cos(theta)]
    ])

if __name__=='__main__':
    real_data = np.load('repel_attract.npy') #open the shape==(600, 417, 2) data
    zlist = np.linspace(0, HEIGHT_OF_TWIST, len(real_data), endpoint=True) # assign appropriate height (in mm)
    thetalist = np.linspace(0, tau, len(real_data), endpoint=False)
    threeDdata = []
    for twoDlist, z, theta in zip(real_data, zlist, thetalist):
        R = rotation_mat(theta)
        x, y = ary([R@point for point in twoDlist]).T
        new_slice = ary([ x, y, np.full(len(x), z) ]).T
        threeDdata.append(new_slice)
    threeDdata = ary(threeDdata)
    np.save('data/3d_twisted.npy', threeDdata)

SCALE_FACTOR = outer_radius_of_cable/radius_max