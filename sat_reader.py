from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
import sys
import time
from numpy.linalg import norm
from cast_into_3d import HEIGHT_OF_TWIST

COLLISION_RAD = 0.035
PLOT = False
with open('matt_input/shortened.sat') as f:
    x1,y1,z1, x2,y2,z2 = ary([[float(i) for i in line.split()] for line in f.readlines()]).T
    collision_mid_pts = (ary([x1+x2, y1+y2, z1+z2])/2).T

with open('matt_input/overlapping-strands.txt') as f:
    strand_pairs = ary([[int(i) for i in line.split(',')] for line in f.readlines()], dtype=int)
    strand_pairs.sort(axis=1)
    strand_pair_set = set([str(i)+','+str(j) for  i,j in strand_pairs])

column = np.load('data/3d_twisted_mat.npy')
reduced_column = column[:,:,[0,1]]

def sort_closest(numpy_ary, scalar):
    order = abs(ary(numpy_ary)-scalar).argsort()
    return order

def sort_closest(numpy_ary, value):
    """
    Find the element that is closest to the value/vector specified.
    Sorted by order of how close they are.
    """
    if numpy_ary.ndim>1:
        return norm(numpy_ary - value, axis=-1).argsort()
    elif numpy_ary.ndim==1:
        return np.argsort(abs(numpy_ary - value))

unit_vec = lambda vec: vec/norm(vec)

if __name__=='__main__':
    zlist = np.linspace(0, HEIGHT_OF_TWIST, len(column), endpoint=True)

    caught_strands = []
    for ind, (x, y, z) in enumerate(collision_mid_pts):
        #closest two slices
        z_nearest, z_next_nearest = sort_closest(zlist, z)[:2]
        strand1_candidates = sort_closest(reduced_column[z_nearest]     , [x,y])[:2]
        strand2_candidates = sort_closest(reduced_column[z_next_nearest], [x,y])[:2]
        for strand in strand1_candidates:
            assert strand in strand2_candidates, "The nearest strands for these two slices ({},{}) does not match! Alternative method is needed.".format(z_nearest, z_next_nearest)
        strand1_candidates.sort()

        caught_strands.append(strand1_candidates)
        disp_vec1 = np.diff([reduced_column[z_nearest     ][strand] for strand in strand1_candidates], axis=0)[0]
        disp_vec2 = np.diff([reduced_column[z_next_nearest][strand] for strand in strand1_candidates], axis=0)[0]
        if PLOT:
            fig, ax = plt.subplots()
            ax.plot( *ary([disp_vec1, disp_vec2]).T )
            ax.scatter(0,0)
            ax.set_aspect(1)
            plt.show()
            plt.close()

        w = disp_vec2 - disp_vec1
        M = [[-w[0], w[1]],[-w[1], -w[0]]]

        beta, n_over_w = np.linalg.solve(M, disp_vec1)
        dist = abs(n_over_w) * unit_vec(w)
        print("distance=",norm(dist), ['AAAAAA' if norm(dist)<COLLISION_RAD else ''])

    
    for i,j in caught_strands:
        strand_str = str(i)+','+str(j)
        try:
            strand_pair_set.remove(strand_str)
        except KeyError as e:
            print(e, "is not in the set of provided strand pairs")
    print('Unmatched strand pair =', strand_pair_set)
