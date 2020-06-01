from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
from SimpleMapping import radius_max
HEIGHT_OF_TWIST = radius_max*10

def rotation_mat(theta):
    return ary([
    [cos(theta), -sin(theta)],
    [sin(theta),  cos(theta)]
    ])

if __name__=='__main__':
    real_data = np.load('data/repel_attract_final.npy')
    zlist = np.linspace(0, HEIGHT_OF_TWIST, len(real_data), endpoint=True)
    thetalist = np.linspace(0, tau, len(real_data), endpoint=False)
    threeDdata = []
    for twoDlist, z, theta in zip(real_data, zlist, thetalist):
        R = rotation_mat(theta)
        x, y = ary([R@point for point in twoDlist]).T
        new_slice = ary([ x, y, np.full(len(x), z) ]).T
        threeDdata.append(new_slice)
    threeDdata = ary(threeDdata)