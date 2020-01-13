from numpy import sign, cos, exp, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.animation as manimation

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
#### basic functions
def get_theta(frame_num):
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
def square_to_sextant(list_of_points, which_sextant=0):
	transformed_list = []
	for [u,v] in list_of_points:
		factor = sqrt( (12*u+pi*offset_constant_in_sqrt/pi)/pi)
		x , y = factor * ary([cos(pi/6*v + pi/3*which_sextant), sin(pi/6*v + pi/3*which_sextant)])
		transformed_list.append([x,y])
	return transformed_list

def circle_to_sextant(list_of_points):
	intermediate_list = elliptical_grid_mapping(list_of_points)
	transformed_list = square_to_sextant(intermediate_list)
	return transformed_list

def rotate(point, theta):
	rotation_matrix = ary([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
	return rotation_matrix @ point

def rotate_list_of_points(list_of_points, theta):
	transformed_list =[]
	for point in list_of_points:
		transformed_list.append( rotate(point, theta) )
	return transformed_list

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

def tessellate_circle_properly(num_sample):
	excess = 0
	while True:
		circle = tessellate_circle(num_sample + excess)
		if len(circle) >= num_sample:
			print(len(circle), "wires are used")
			return circle
		excess += 1
####

#### basic transformation into 3D workflow
def get_z_of_central_column(theta):
	z = h*theta
	return z

def transform_2D_into_3D(list_of_points, z_of_central_column):
	list_of_3D_points = []
	return list_of_3D_points
####
if __name__=="__main__":
	list_of_coordinates = tessellate_circle(450)#(wire_per_sextant)
	'''
	step_size = 2*pi/num_frames

	# initializing the animation frame
	
	FFMpegWriter = manimation.writers['ffmpeg']
	metadata = dict(title='MappingCheck', artist='Matplotlib', comment='For examining optimal mappings')
	writer = FFMpegWriter(fps=15, metadata=metadata)
	fig, ax = plt.subplots()
	ax.set_aspect(1.0)
	ax.set_xlim(-1, 1)
	ax.set_ylim(-1, 1)
	ax.set_xticks([])
	ax.set_yticks([])
	# initializing outline and data
	for sex in range(1):
		c = 'C'+str(sex)
		center = rotate([sqrt(offset_constant_in_sqrt/pi),0], sex*angle_between_sextant)
		x_orig, y_orig = ary([ center for i in range(len(list_of_coordinates)) ]).T
		scat = ax.scatter(x_orig, y_orig, facecolor='white', edgecolor=c, linewidths=1.0, s=3.0**2)
	#The dotted line represents the direction in the cross-section view which is parallel to (i.e. in the) the xy plane.
	#make the scatter plot
	with writer.saving(fig, "MappingCheck.mp4", num_frames):
		for frame in range(num_frames):
			theta = get_theta(frame)
			ax.set_title( r'$\theta=${0:.2f}$^o$'.format( round(np.rad2deg(theta+pi/2),2)%360 ) )
			#sextant dependent tasks
			for sex in range(1):
				rotated_circle = rotate_list_of_points(list_of_coordinates, theta)
				transformed_list = elliptical_grid_mapping(rotated_circle)
				scat.set_offsets( transformed_list )
			writer.grab_frame()
		fig.clf()
	'''