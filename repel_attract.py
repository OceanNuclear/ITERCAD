from numpy import array as ary
from numpy import log   as ln
from numpy import exp, sqrt
from numpy import pi, arcsin, tan; tau = 2*pi
import numpy as np
from SimpleMapping import tessellate_circle_properly, rotate_list_of_points, circle_to_sextant, radius_max, radius_min
import time
from scipy.stats import describe

RESOLUTION = 600
diameter = 0.035886910451285364*2
EFF_RADIUS = 0.15
MAX_REF_STEP_SIZE = diameter*1 # effective radius

def calc_dist(frame):
    """
    In the fastest/most compact way, calculate the distances in the frame.
    """
    return sqrt(frame.reshape([-1,2])[:,0]**2 + frame.reshape([-1,2])[:,1]**2).reshape(frame.shape[:-1])
    # a slower but clearer implementation is listed here:
    # flat_frame = frame.reshape([-1,2])
    # dists = sqrt(flat_frame[:,0]**2 + flat_frame[:,1]**2)
    # return dists.reshape(list(frame.shape)[:-1])

def calc_tan_ang(positions):
    """
    Calculate the tangent of the angle of the position of the wire from the x-axis
    """
    return (positions.reshape([-1,2])[:,1]/positions.reshape([-1,2])[:,0]).reshape(positions.shape[:-1])

def project_onto_vector(positions, unit_vector):
    """
    Project the point onto the specified unit_vector
    """
    return (unit_vector[0] * positions.reshape([-1,2])[:,0] + unit_vector[1] * positions.reshape([-1,2])[:,1]).reshape(positions.shape[:-1])

def get_unit_vec(positions):
    """
    Get the unit vector pointing from the origin to the direction of the point
    """
    return positions/np.repeat(calc_dist(positions), 2, axis=-1).reshape(positions.shape)

def mean_vel_vector(*vel_vectors):
    """
    Find the average motion for each point,
    after the force vectors are given.
    """
    num_slice, num_points = vel_vectors[0].shape[:2]
    assert_msg = "They must all have the same number of slices and points per slice"
    assert all([ vel_vec.shape[:2]==(num_slice, num_points) for vel_vec in vel_vectors]), assert_msg
    return np.nan_to_num([[ np.concatenate([vel_vec[n,m].reshape([-1,2]) for vel_vec in vel_vectors], axis=0).mean(axis=0)
                    for m in range(num_points) ]
                        for n in range(num_slice)],
                nan=0) # triple nested loop

def get_disp_vec(column, f_source):
    """
    List of displacement vector between different points.
    disp_vec[n][o][m][x_or_y]
    n: index of slice
    o: index of point experiencing the force
    m: index of point creating the force
    x_or_y: takes the value of 0 or 1 respectively.
    In other words, disp_vec[n][o][m] shows the vector that points from point[m] to point[o] (in slice n)
    np.mean(disp_vec[n][o,:], axis=0) gives the average vector pointing towards point o 
            (i.e. average force acting on point[o]) in slice n.
    """
    num_slice, num_points, _ = column.shape
    # Find the Δx and Δy of each point
    disp_vec = [ [ ary([target[m, 0] - source[:, 0],    # Δx
                        target[m, 1] - source[:, 1]]).T # Δy
                            for m in range(len(source))]
                                for target, source in zip(column, f_source)] # In each slice,
    return ary(disp_vec)
    # slower but clearer implementation
    # Δx = ary([column[:, m, 0] - f_source[:,:, 0].T for m in range(num_points)])
    # Δy = ary([column[:, m, 1] - f_source[:,:, 1].T for m in range(num_points)])
    # return ary([Δx.T.reshape([-1]), # flattening out the Δx and Δy takes a long time
    #             Δy.T.reshape([-1])]).T.reshape([num_slice, num_points, num_points, 2])
    # return ary([ ary([
    #     column[:, m, 0] - f_source[:,:, 0].T,# Δx
    #     column[:, m, 1] - f_source[:,:, 1].T # Δy
    # ]) for m in range(num_points) ]) #m denotes the number of points

def repel(column):
    """
    Find the velocity contributed by all within the same layer
    """
    disp_vec = get_disp_vec(column, column)
    dists = calc_dist(disp_vec) # boolean dists

    return disp_vec * np.nan_to_num(np.repeat(
                np.clip(EFF_RADIUS/dists -1, 0, None), 2, axis=-1
                ), nan=0, posinf=0, neginf=0).reshape(disp_vec.shape)

def repel_dense(target, source, RAD=EFF_RADIUS):
    """
    Same as above, but outputs in dense represenation. It's faster this way.
    """
    disp_vec = get_disp_vec(target, source)
    dists = calc_dist(disp_vec) # boolean dists
    mask = dists < RAD # boolean adjacency matrix # the mask shouldn't be used at this stage yet,

    vel_vec = []
    for i, this_slice in enumerate(disp_vec): # each 'this_slice' is an array of shape (num_points, num_points, 2)
        np.fill_diagonal(mask[i], False) # ignore the original point itself (i.e. the main diagonal)
        # The formula for computing the disp_vec is
        # max(r/d-1,0) * (displacement vector)
        vel_vec.append( this_slice[mask[i]] * np.repeat(np.clip(RAD/dists[i][mask[i]] -1, 0, None), 2).reshape([-1,2]) )
    return apply_mask_on_sparse(np.nan_to_num(vel_vec, nan=0, posinf=0, neginf=0), mask)

def apply_mask_on_sparse(vel_vector, mask):
    """
    Given the mask and the sparse representation of ALL velocity vectors,
    apply the mask so that the vel_vector collection changes from
    (600, 417, 417)
    #^ slice number
    #     ^ force acts on this point
    #          ^ force comes from this point
    to
    (600, 417, variable-length)
    """
    cumsumed_mask = np.cumsum(mask.sum(axis=-1), axis=1).astype(int) # sum the last axis, i.e. the axis representing the points from f_source.
    vel_vec_out = []
    for vel_vec, mask_line in zip(vel_vector, cumsumed_mask): #iterate through all slices
        vel_vec_out.append(
            ary([vel_vec[start:stop] for (start, stop) in zip(
                [0]+mask_line[:-1].tolist(), mask_line
                )])
        )
    return ary(vel_vec_out)

def apply_mask_on_custom(vel_vec, mask):
    """
    No docstring
    """
    vel_vec_out = []
    for this_slice, mask_line in zip(vel_vec, mask):
        vel_vec_out.append([ vector if required else ary([[],[]]).T for vector, required in zip(this_slice, mask_line)])
    return ary(vel_vec_out)

def populate_with_vector(mask, vector):
    """
    Fill in a vector if applicable ('True' in mask)
    2x run of this function (which involves the copying operation)
    using real_data (600x417 points) took only 0.8 seconds.
    so it was deemed that the copying doesn't introduce much inefficency
    and thus is not a point to be optimized.
    """
    vec = ary(vector.copy())
    return ary( [[vec.copy() if required else ary([[],[]]).T for required in mask_line] for mask_line in mask] )

def wall_repel(column):
    """
    The force supplied by the walls onto the values.
    Need to perform checks
    """
    dists = calc_dist(column) # calculate the distance of the point from the origin
    # inner_wall = get_unit_vec(column) * np.clip((radius_min + EFF_RADIUS - dists), 0, None) # only care about the one that are positive,
    # i.e. those that undershoot the inner wall radius.
    # outer_wall = get_unit_vec(column) * np.clip((radius_max - EFF_RADIUS - dists), None, 0) #only care about the ones that are negative,
    # i.e. those that overshoot the outer wall radius.
    inner_wall = apply_mask_on_custom( get_unit_vec(column), dists<(radius_min + EFF_RADIUS) ) * (radius_min + EFF_RADIUS - dists)
    outer_wall = apply_mask_on_custom( get_unit_vec(column), dists>(radius_max - EFF_RADIUS) ) * (radius_max - EFF_RADIUS - dists)

    upper_wall_dist = project_onto_vector(column, unit_vector=ary([-1,2])/sqrt(5))
    lower_wall_dist = project_onto_vector(column, unit_vector=ary([ 1,2])/sqrt(5))
    upper_wall = (EFF_RADIUS + upper_wall_dist) * populate_with_vector(upper_wall_dist > -EFF_RADIUS, [ 1/sqrt(5),-2/sqrt(5)])
    # upper_wall_dist: (-C,0]
    # EFF_RADIUS - upper_wall_dist: [EFF_RADIUS,-C+EFF_RADIUS)
    # (EFF_RADIUS - upper_wall_dist)[upper_wall_dist+EFF_RADIUS>0]: [EFF_RADIUS, 0]
    lower_wall = (EFF_RADIUS - lower_wall_dist) * populate_with_vector(lower_wall_dist <  EFF_RADIUS, [ 1/sqrt(5), 2/sqrt(5)])
    return inner_wall, outer_wall, upper_wall, lower_wall

def attract(column, height):
    """
    Find the velocity vectors contributed
    by the attraction of each point
    by the corresponding point to the slice above itself.
    """
    return ary(column-np.roll(column,-height, axis=0))
     # using s-t instead of t-s since we're trying to do attraction instead.

def create_test_data(random=False):
    """Create extremely small dataset which can be printed"""
    if random:
        np.random.seed(0)
        data = np.random.uniform(size=24).reshape([4,3,2]) # create a slice of 
    else:
        data = np.arange(24).reshape([4,3,2])
    return data

def lay_out_vel_vecs(forces):
    """
    return lay_out_vel_vecs,
    plus the max number of vectors acting on any point
    """
    return ary([[vel_vecs_on_pt.sum(axis=0) for vel_vecs_on_pt in this_slice] for this_slice in forces]), max([len(a) for a in forces.flatten()])

def lay_out_wall_vel_vecs(this_wall_vec):
    """
    same as above 
    """
    return ary([ary([vec.copy() if len(vec) else ary([0,0]) for vec in wall_rep.flatten()]) for wall_rep in this_wall_vec]).reshape([*this_wall_vec.shape, 2])

if __name__=='__main__':
    starttime=time.time()
    # circle = tessellate_circle_properly(417)
    # real_data = []
    # for theta in np.linspace(0, tau, RESOLUTION):
    #     rotated_circle = rotate_list_of_points(circle, theta)
    #     transformed_list = circle_to_sextant(rotated_circle)
    #     real_data.append(transformed_list)
    # real_data = ary(real_data)
    # np.save('PRESTINE.npy', real_data)

    real_data = np.load('PRESTINE.npy')
    # data = create_test_data()
    num_steps = 0
    print('Starting at time', time.time()-starttime, 's')

    while (num_steps:=num_steps+1):
        self_repel = repel_dense(real_data, real_data)
        upper_repel = repel_dense(real_data, np.roll(real_data, -1, axis=0), RAD=EFF_RADIUS*0.95)
        lower_repel = repel_dense(real_data, np.roll(real_data,  1, axis=0), RAD=EFF_RADIUS*0.95)
        wall_vel_vecs = wall_repel(real_data)

        two_above = attract(real_data, 2)
        one_above = attract(real_data, 1)
        one_below = attract(real_data, -1)
        two_below = attract(real_data, -2)

        final_velocity, weights = [], []
        for forces in (self_repel, upper_repel, lower_repel):
            sum_vel, max_vecs = lay_out_vel_vecs(forces)
            weights.append(max_vecs)
            final_velocity.append(sum_vel)
        for ind, (vel, max_vecs) in enumerate(zip(final_velocity, weights)):
            real_data += vel * max_vecs/sum(weights) * 0.4

        for this_wall_vec in wall_vel_vecs:
            real_data += lay_out_wall_vel_vecs(this_wall_vec)*0.4 #amplify the forces from the walls back to 1 instead of 0.25
        real_data += ((one_above+one_below)/2 + (two_above+two_below))* 0.05

        # average_movement = mean_vel_vector(self_repel, upper_repel, lower_repel, upper_attract, lower_attract, *wall_vel_vecs)
        # real_data += average_movement

        print( 'Taken step={} at time={}s'.format(num_steps, round(time.time()-starttime,2) ) )
        np.save('repel_attract.npy', real_data)

        # # printing information:
        # print(describe(lower_attract[0]))
        # print(describe(upper_attract[0]))
        # print(describe(upper_attract[0]))

        # # breaking
        if num_steps == 24:
            break