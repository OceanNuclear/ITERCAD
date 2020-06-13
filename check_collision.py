from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
import sys
from repel_attract import calc_dist, get_unit_vec, get_disp_vec, repel
import time
from numpy.linalg import inv, norm
from collections import defaultdict
from SimpleMapping import radius_min, radius_max

area = (radius_max**2 - radius_min**2)*pi/6
absolute_max_area_per_wire = area/417 # absolute maximum cross-section of each wire if they can be deformed
square_packing_max_area = absolute_max_area_per_wire*pi/4
area_assumed = square_packing_max_area * 0.4
radius_assumed = sqrt(area_assumed/pi)
DIAMETER = radius_assumed*2

if __name__=='__main__':
    # print("Same as above, but assuming packing factor = 0.5, is", 2*sqrt(absolute_max_area_per_wire*0.5/pi))
    print("Maximum diameter actually used is", DIAMETER)

    # x,y = np.load(sys.argv[-1]).T
    x,y = np.load(sys.argv[-1]).T[:2]
    # x,y,z = np.load('data/3d_twisted.npy').T
    column = ary([x,y]).T
    num_slice, num_points, _ = column.shape
    """
    Calculate the closest the o-th wire gets to the m-th wire at the n-th slice
    """
    # for the m-th target strand,
    dist_at_interslice_spacing = np.zeros([num_slice, num_points, num_points])
    for s in range(num_slice):
        np.fill_diagonal(dist_at_interslice_spacing[s], radius_max)
    num_parallels = np.zeros([num_slice, num_points])
    # dist_at_interslice_spacing[n, m] refers to the closest any wire gets to wire m between slice n and n+1
    starttime = time.time()
    all_min_dists = []
    for m in range(num_points):
        collective_dist = [[],]*num_slice
        vec_u_x = column[:,:,0].T - column[:, m, 0]
        vec_u_y = column[:,:,1].T - column[:, m, 1]
        vec_U = ary([vec_u_x, vec_u_y]).T
        vec_W = np.roll(vec_U, -1, axis=0) - vec_U
        len_U = norm(vec_U, axis=-1)
        len_W = norm(vec_W, axis=-1)
        # calculate the minium distances at each slice
        W0 = vec_W[:,:,0].T
        W1 = vec_W[:,:,1].T
        M = ary([-W0,W1,-W1,-W0]).T.reshape([*vec_W.shape[:-1], 2,2])
        M[:,m] = np.identity(2)

        beta, n_over_w = [array.T for array in np.linalg.solve(M, vec_U).T]

        dist = len_W * abs(n_over_w)
        perpendicular = np.logical_and(0<=beta, beta<=1)
        perpendicular[:,m] = False # exclude the point itself
        below = beta<0
        above = beta>1
        dist_at_interslice_spacing[perpendicular, m] = dist[perpendicular].copy()
        dist_at_interslice_spacing[below, m] = len_U[below].copy()
        dist_at_interslice_spacing[above, m] = np.roll(len_U, -1, axis=0)[above].copy()
        # for lineno, boolean_mask in enumerate(perpendicular):
        #     for i in dist[lineno, boolean_mask]:
        #         collective_dist[lineno].append(i)
        #     for i in len_U[lineno, below[lineno]]:
        #         collective_dist[lineno].append(i)
        #     for i in len_U[(lineno+1)%num_slice, above[lineno]]:
        #         collective_dist[lineno].append(i)
        # min_dist_for_strand_n = np.min(collective_dist, axis=1)
        # all_min_dists.append(min_dist_for_strand_n)

        """
        for n in range(num_slice):
            distances = []
            in_range_beta = 0
            for o in [o for o in range(num_points) if o!=m]: # skip the m-th wire itsel itself
                w = vec_W[n, o]
                M = [[-w[0], w[1]], [-w[1], -w[0]]]
                beta, n_over_w = np.linalg.solve(M, vec_U[n, o])
                if 0<=beta<=1:
                    distances.append(abs(n_over_w) * np.linalg.norm(w))
                    in_range_beta += 1
                elif beta<0:
                    distances.append(np.linalg.norm(vec_U[n,o]))
                elif beta>1:
                    distances.append(np.linalg.norm(vec_U[(n+1)%num_slice,o]))
            dist_at_interslice_spacing[n, m] = min(distances).copy()
            num_parallels[n, m] = in_range_beta
            assert len(distances)==(num_points-1), f"expected {num_points-1} points' distances to be calculated"
        """
        if True:
            print('Finished strand {} at time={} s. Expected progress {}/{} minutes'.format(m+1,
                    round(time.time()-starttime,2),
                    round((time.time()-starttime)/60),
                    round((time.time()-starttime)/60*num_points/(m+1))
                        ),
                    end='\r')
    print(f'Displaying all places in the model where wires are closer than {DIAMETER}:')
    for collision_ind, (slice_ind, strand1ind, strand2ind) in enumerate(np.argwhere(dist_at_interslice_spacing<DIAMETER)):
        print("{:2}:At inter-slice space {:4} (between {:4} and {:4}), found collision between strand {:3}-{:3}. They have a spacing of {}".format(collision_ind//2, slice_ind, slice_ind, (slice_ind+1)%num_slice, strand1ind, strand2ind, dist_at_interslice_spacing[slice_ind, strand1ind, strand2ind]))