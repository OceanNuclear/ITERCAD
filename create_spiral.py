from numpy import sign, cos, exp, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#### parameters
target_r_min_to_r_max_ratio = 0.45/1.985 #both radii measurements were recorded in mm.
offset_constant_in_sqrt = 12 * (1 + target_r_min_to_r_max_ratio**2)/(1 - target_r_min_to_r_max_ratio**2)
radius_max = sqrt(offset_constant_in_sqrt+12)/sqrt(pi)
radius_min = sqrt(offset_constant_in_sqrt-12)/sqrt(pi)
wire_per_sextant = int(np.floor(2500/6))
mapping = 'elliptical_grid_mapping'
num_frames = 300
h = 10 #height required to go up in order to complete one rotation (in mm) 
TILT = arctan(1/h)
angle_between_sextant = pi/3 #created for convenience
aesthetic_offset_angle = angle_between_sextant

#### basic functions
def get_theta(frame_num):
    '''Translate frame number back into angle corresponding to that frame of the video'''
    return tau*frame_num/num_frames

def quadrature(coordinates_list):
    return sqrt(sum([i**2 for i in coordinates_list]))

def get_radius_given_number(number_in_circle):
    # asymptotic limit of number of circles that can be hexagonally packed in a square.
    equivalent_number_in_square = 4/pi * number_in_circle
    return 2/sqrt(equivalent_number_in_square*2*sqrt(3))
####

#### mappings
def elliptical_grid_mapping(list_of_points):
    transformed_list = []
    for [u,v] in list_of_points:
        x = 1.0/2 * (sqrt(2+u**2-v**2+2*sqrt(2)*u) - sqrt(2+u**2-v**2-2*sqrt(2)*u))
        y = 1.0/2 * (sqrt(2-u**2+v**2+2*sqrt(2)*v) - sqrt(2-u**2+v**2-2*sqrt(2)*v))
        transformed_list.append([x,y])
    return transformed_list

def simple_stretch(list_of_points):
    transformed_list = []
    for [u,v] in list_of_points:
        u2, v2 = u**2, v**2
        if u2>=v2:
            x = sign(u)*sqrt(u2+v2)
            y = sign(u)*sqrt(u2+v2)*v/u
        else:
            x = sign(v)*sqrt(u2+v2)*u/v
            y = sign(v)*sqrt(u2+v2)
        transformed_list.append([x,y])
    return transformed_list

def FG_squircular(list_of_points):
    transformed_list = []
    for [u,v] in list_of_points:
        u2, v2 = u**2, v**2
        if u2+v2 - sqrt( (u2+v2) * (u2-4*u2*v2+v2) ) == 0:
            x, y = u, v
        else:
            x = sign(u*v)/(v*sqrt(2)) * sqrt(u2+v2 - sqrt( (u2+v2) * (u2-4*u2*v2+v2) ) )
            y = sign(u*v)/(u*sqrt(2)) * sqrt(u2+v2 - sqrt( (u2+v2) * (u2-4*u2*v2+v2) ) )
        transformed_list.append([x,y])
    return transformed_list

# from scipy.special import ellipkinc
def Schwarz_Christoffel(list_of_points):
    transformed_list = []
    norm11 = complex(1,1)/sqrt(2)
    Ke = ellipkinc(pi/2, 1/2)
    for [u,v] in list_of_points:
        w = complex(u,v)
        x = (complex(1,-1)/(-Ke) * ellipkinc( arccos(norm11*w), 1/sqrt(2)) ).real + 1
        y = (complex(1,-1)/(-Ke) * ellipkinc( arccos(norm11*w), 1/sqrt(2)) ).imag - 1
        transformed_list.append([x,y])
    return transformed_list
####

#### basic 2D mapping and rotation workflow programs
def square_to_sexant(list_of_points, which_sextant=0):
    transformed_list = []
    for [u,v] in list_of_points:
        factor = sqrt( (12*u+pi*offset_constant_in_sqrt/pi)/pi)
        x , y = factor * ary([cos(pi/6*v + pi/3*which_sextant), sin(pi/6*v + pi/3*which_sextant)])
        transformed_list.append([x,y])
    return transformed_list

def circle_to_sextant(list_of_points):
    intermediate_list = elliptical_grid_mapping(list_of_points) # choose the elliptical grid mapping method
    transformed_list = square_to_sexant(intermediate_list)
    return transformed_list

def rotate(point, theta):
    rotation_matrix = ary([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return rotation_matrix @ point

def rotate_list_of_points(list_of_points, theta):
    transformed_list =[]
    for point in list_of_points:
        transformed_list.append( rotate(point, theta) )
    return transformed_list

def get_uniformly_covered_circle(num_sample):
    output_list = []
    while len(output_list)<num_sample:
        if quadrature(point := np.random.uniform(-1, 1, [2]))<=1:
            output_list.append(point)
    return output_list

def tessellate_circle(num_sample):
    #create hexagonal packing of circles:
    tessellated_square = ary([1]) #create dummy
    mask = [False] #create dummy
    try_number = num_sample - 1
    while len(tessellated_square[mask])<num_sample:
        try_number+=1
        r = get_radius_given_number(try_number)
        xspace = 2*r
        yspace = 2*sqrt(3)*r
        full_rows = ary(np.meshgrid(np.arange(-1+r, 1, xspace), np.arange(-1+r, 1, yspace))).T.reshape([-1,2])
        in_between_rows = ary(np.meshgrid(np.arange(-1+2*r, 1, xspace), np.arange(-1+(1+sqrt(3))*r, 1, yspace))).T.reshape([-1,2])
        tessellated_square = np.concatenate([full_rows, in_between_rows])
        mask = [ quadrature(point)<1-0.8*r for point in tessellated_square]
    return tessellated_square[mask]
####

#### basic transformation into 3D workflow
def get_z_of_central_column(theta):
    z = h*theta
    return z

def Rotation2Quat(axis, angle):
    '''Function to turn (axis-angle) representation of a rotation operatoin into a quaternion representation'''
    quat = [cos(angle/2), sin(angle/2)*axis[0], sin(angle/2)*axis[1], sin(angle/2)*axis[2]]
    return quat

def Quat2R(q):
    '''Reusing quaternion functions for simulating rotation'''
    theta = 2 * arccos(np.clip(q[0],-1,1))

    R = np.identity(3)
    if theta>2E-5 or abs(theta-pi)>2E-5: # theta_prime not approaching 0 or pi
        axis = q[1:]/sin(theta/2)

        R[0][0] -= 2*( q[2]**2  +q[3]**2  )
        R[1][1] -= 2*( q[1]**2  +q[3]**2  )
        R[2][2] -= 2*( q[1]**2  +q[2]**2  )
        R[0][1] -= 2*( q[0]*q[3]-q[1]*q[2])
        R[0][2] -= 2*(-q[0]*q[2]-q[1]*q[3])
        R[1][0] -= 2*(-q[0]*q[3]-q[1]*q[2])
        R[1][2] -= 2*( q[0]*q[1]-q[2]*q[3])
        R[2][0] -= 2*( q[0]*q[2]-q[1]*q[3])
        R[2][1] -= 2*(-q[0]*q[1]-q[2]*q[3])

    return R

def rotate_list_of_2D_points_into_3D(R_matrix, list_of_points, z=0):
    output_3D_list = []
    for [u,v] in list_of_points:
        output_3D_list.append(R_matrix@[u,v,0]+ary([0,0,z]))
    return output_3D_list
####

def calculate_distances(list_of_points):
    distances_matrix = []
    for point in list_of_points:
        distance_row = ary(list_of_points)-point
        distances_matrix.append([ quadrature(dist) for dist in distance_row ])
    return distances_matrix

def get_closest_neighbour(distances_matrix):
    copy_matrix = ary(distances_matrix.copy())
    max_val = copy_matrix.max()
    copy_matrix += np.diag( np.ones(len(copy_matrix))*max_val )  # bodge to add some values onto the main diag
    return copy_matrix.argmin(axis=1)

def get_closest_neighbour_distance(list_of_coordinates):
    distances_matrix = calculate_distances(list_of_coordinates)
    closest_neighbour_index = get_closest_neighbour( distances_matrix )
    column_of_neighbours = []
    for i in range(len(list_of_coordinates)):
        column_of_neighbours.append(distances_matrix[i][closest_neighbour_index[i]])
    return column_of_neighbours

if __name__=="__main__":
    list_of_coordinates = tessellate_circle(250)
    # list_of_coordinates = [[1,0],[0,0],[0,1]]

    step_size = 2*pi/num_frames

    # initializing outline and data
    outline_template = [None,]*6
    # subsheath = {i:[] for i in range(6)}
    bundle = {i:[] for i in range(6)} # records the coordinates at the current frame only

    for sex in range(6):
        #get the cable sheath
        '''
        outline_resolution = 300
        outline_orig = [ rotate(ary([1,0]), tau*fraction_of_circle/outline_resolution ) for fraction_of_circle in range(outline_resolution+1) ]
        output_outline = circle_to_sextant(outline_orig)
        outline_template[sex] = rotate_list_of_points(output_outline, sex*angle_between_sextant) # [[x1,y1],[x2,y2],[x3,y3]...]
        '''
        #initialize the dummy variables for the starting points of all cables
    unit_vector = [1,0]
    angle_of_rotation = -TILT
    #The dotted line represents the direction in the cross-section view which is parallel to (i.e. in the) the xy plane.
    #make the scatter plot
    for frame in range(num_frames):
        theta = get_theta(frame)
        z = get_z_of_central_column(theta)
        axis = rotate(unit_vector, theta)
        R = Quat2R(Rotation2Quat([*axis,0], angle_of_rotation))
        # # ax.set_title( r'$\theta=${0:.2f}$^o$ at $\varphi=${1:.2f}$^o$'.format(round(np.rad2deg(TILT),2), round(np.rad2deg(theta+pi/2),2)%360) )
        #sextant dependent tasks
        for sex in range(6):
            #get the circle
            rotated_circle = rotate_list_of_points(list_of_coordinates, theta - sex*aesthetic_offset_angle) # rotate it appropriately according to theta #make each sextant look slightly different using sex*angle_between_sextant
            transformed_list = circle_to_sextant(rotated_circle)
            rotated_transformed_2D_list = rotate_list_of_points(transformed_list, sex*angle_between_sextant + theta)
            #print the smallest non-trivial distance
            if sex==0:
                neighbour_distance = ary(get_closest_neighbour_distance(rotated_transformed_2D_list))
                print(f"{neighbour_distance.min()=}, {neighbour_distance.max()=}")

            #turn it into 3D
            tilted_into_3D = rotate_list_of_2D_points_into_3D(R, rotated_transformed_2D_list, z)
            bundle[sex].append(tilted_into_3D)
            '''
            subsequent bodge to rotate the WHOLE circle
            rotated_outline = rotate_list_of_points( outline_template[sex], theta )
            tilted_outline = rotate_list_of_2D_points_into_3D(R, rotated_outline, z)
            subsheath[sex].append(tilted_outline)
            '''
def get_spirals_accurately(height_per_twist=h, resolution=num_frames):
    list_of_coordinates = tessellated_square(450)
    step_size = 2*pi/num_frames
    outline_resolution = 300

    #create empty variables to hold numbers
    bundle_sorted_by_strand = {i:[] for i in range(6)}

    for sex in range(6):
        for strand in list_of_coordinates:
            coordinate_tracker = generate_a_single_strand_accurately(sex, strand, height_per_twist=height_per_twist, resolution=resolution)
            bundle_sorted_by_strand[sex].append(coordinate_tracker)
    return bundle_sorted_by_strand # bundle_sorted_by_strand[0<=sex<=5][0<=strand<=416][0<=frame_number<=resolution-1][xyz]

'''
The data is then stored in the following order
bundle[sextant_number][frame_number][strand_number] => [x,y,z]
subsheath[sextant_number][frame_number][point along the subsheath]=>[x,y,z]
'''

#This is parametrisable as a function of z of the CENTRAL column, not of the point itself
def generate_a_single_strand(sextant_number, starting_strand_position, height_per_twist = h, resolution=num_frames):
    #assume rotational speed inside subcable = 2 pi per rotation = rotational speed of sub-cable around central column
    assert np.shape(starting_strand_position)==(2,), "strand starts at a 2D coordinate"
    assert quadrature(starting_strand_position)<=1, "strand must start at a location inside the unit circle"
    coordinate_tracker = []
    for step in range(resolution):
        theta = tau * step/resolution
        height = height_per_twist * step/resolution
        rotated_strand_position = rotate(starting_strand_position, theta - sextant_number*aesthetic_offset_angle) #rotate by theta within the subcble
        transformed_point = circle_to_sextant([rotated_strand_position,])[0] #pack into list and then unpack
        rotated_point = rotate(transformed_point, theta + sextant_number*angle_between_sextant)

        axis = rotate([1,0], theta)
        R = Quat2R(Rotation2Quat([*axis,0], -arctan(1/h)))

        coordinate_tracker.append( R @ [*rotated_point,0] + ary([0,0,height]) )
    return coordinate_tracker

def generate_a_single_strand_accurately(sextant_number, starting_strand_position, height_per_twist = h, resolution=num_frames):
    #assume rotational speed inside subcable = 2 pi per rotation = rotational speed of sub-cable around central column
    assert np.shape(starting_strand_position)==(2,), "strand starts at a 2D coordinate"
    assert quadrature(starting_strand_position)<=1, "strand must start at a location inside the unit circle"
    coordinate_tracker = []
    for step in range(resolution):
        theta = tau * step/resolution
        height = height_per_twist * step/resolution
        rotated_strand_position = rotate(starting_strand_position, theta - sextant_number*aesthetic_offset_angle) #rotate by theta within the subcble
        transformed_point = circle_to_sextant([rotated_strand_position,])[0] #pack into list and then unpack
        # rotated_point = rotate(transformed_point, theta + sextant_number*angle_between_sextant)

        axis1 = rotate([1,0], theta)
        R1 = Quat2R(Rotation2Quat([*axis1,0], -arctan(1/h)))

        tilted_point = R1@[*transformed_point, 0] + ary([0,0, height])

        axis2 = [0,0,1]
        R2 = Quat2R(Rotation2Quat(axis2, theta + sextant_number*angle_between_sextant))

        coordinate_tracker.append( R2 @ tilted_point )
    return coordinate_tracker

def parametrised_as_a_function_of_height(sextant_number, starting_strand_position, vertical_position, height_per_twist=h):
    assert np.shape(starting_strand_position)==(2,), "strand starts at a 2D coordinate"
    assert quadrature(starting_strand_position)<=1, "strand must start at a location inside the unit circle"
    theta = tau * ((vertical_position/height_per_twist)%1.0)
    rotated_strand_position = rotate(starting_strand_position, theta - sextant_number*aesthetic_offset_angle) #rotate by theta within the subcble
    transformed_point = circle_to_sextant([rotated_strand_position,])[0] #pack into list and then unpack
    rotated_point = rotate(transformed_point, theta + sextant_number*angle_between_sextant)

    axis = rotate([1,0], theta)
    R = Quat2R(Rotation2Quat([*axis,0], -arctan(1/h)))    
    
    return R @ [*rotated_point,0] + ary([0,0,vertical_position])