import numpy as np
from numpy import array as ary
from numpy import sqrt, tan, pi, sin, cos
from math import fsum
import os
from SimpleMapping import *
#import seaborn as sns
from scipy.stats import describe
from time import time
import logging

#given array of points as all_points
RAD = 0.15
quadrature = lambda lst : sqrt(sum([i**2 for i in lst]))
REF_STEP_SIZE = 1 
RESOLUTION = 1200
#determine when to stop iterating.
area_sextant = pi*(radius_max**2 - radius_min**2)/6
wire_radius = sqrt(area_sextant/417)/2 # this is only an approximation, assuming square packing.
F_LIMIT = wire_radius * 0.5E-2 # 
NEW_KERNEL = True

RUNNING_ON_LOCAL = 'ocean' in os.uname()[1]
if RUNNING_ON_LOCAL:
	import matplotlib.pyplot as plt

if NEW_KERNEL:
	def kernel(scaled_dist):
		return 1- scaled_dist

	PARABOLIC_PART_WEIGHT = 0.5
	def kernel_2(scaled_dist):
		return 1- scaled_dist + PARABOLIC_PART_WEIGHT*(1- scaled_dist)**2
		
	def inv_kernel(scaled_dist):
		return 1-scaled_dist

	def inv_kernel_2(y):
		return ( (1/PARABOLIC_PART_WEIGHT +2) - sqrt( (1/PARABOLIC_PART_WEIGHT + 2)**2 - 4*(1/PARABOLIC_PART_WEIGHT+1-y) ) )/2
else:
	def kernel(scaled_dist):
		scale_fac = pi/2 #scale to the right radius
		return 1/scale_fac * 1/tan(scaled_dist*scale_fac) # cot(number/(pi/2) ), the 1/scale_fac is intended to keep the slope =-1.

	def inv_kernel(y): # This function seems wrong...
		return pi/2*tan(pi/2*y)

class Point:
	def __init__(self, pos, pk=kernel, wk=kernel):
		self.pos = ary(pos) # should be a list of len=2
		self.radius_of_effectiveness = RAD
		self.wall_effect_thickness = RAD
		self.pk_raw = pk
		self.wk_raw = wk
		#self.closest_point = self.radius_of_effectiveness.copy() #for tracking where is the nearest particle
		
	def point_kernel(self, dist):
		scaled_dist = dist/self.radius_of_effectiveness
		return self.radius_of_effectiveness* self.pk_raw(scaled_dist) #again, these are meant to keep its slope = -1
		
	def wall_kernel(self, dist):
		scaled_dist = dist/self.wall_effect_thickness
		return self.wall_effect_thickness * self.wk_raw(scaled_dist) #again, these are meant to keep its slope = -1
		
	def get_force_soph(self, all_points, i):
		force = []
		
		mask = np.ones(len(all_points), dtype=bool)
		mask[i] = False
		for p in ary(all_points)[mask]: # have to convert it into an array in order to work with mask methods.
			displacement = self.pos - p.pos
			dist = quadrature(displacement)

			if dist < self.radius_of_effectiveness:
				direc = displacement/dist
				mag = self.point_kernel(dist)
				force.append(mag*direc)
		return ary(force).reshape([-1,2])

	def get_force_raw(self, all_points):
		force = []
		wall_force = self.wall_repel()
		if quadrature(wall_force): # if the force is non-zero:
			force.append(wall_force)
		
		for p in all_points:
			if not all(p.pos== self.pos): # for all points that isn't itself:
				displacement = self.pos - p.pos	# calculate the difference
				dist = quadrature(displacement)
			
				if dist < self.radius_of_effectiveness: # if the point is close enough:
					direc = displacement/dist # normalize the negative displacement vector
					mag = self.point_kernel(dist)
					force.append(mag*direc)
		return ary(force).reshape([-1,2]) #ary([[x,y],[x,y],...])

	def get_force(self, all_points):
		force = self.get_force_raw(all_points)
		averaged_force = [ sum(force[:,0])/len(force), sum(force[:,1])/len(force) ]
		return averaged_force

	def wall_repel(self):
		force = ary([0., 0.])
		# tangential walls' repulsion
		if quadrature(self.pos) < radius_min + self.wall_effect_thickness: # close to the centre of the cable
			direc = self.pos/quadrature(self.pos)
			dist = quadrature(self.pos) - radius_min
			force += direc* self.wall_kernel(dist) #distance from the inner radius wall
		elif quadrature(self.pos) > radius_max - self.wall_effect_thickness:
			direc = -self.pos/quadrature(self.pos)
			dist = radius_max - quadrature(self.pos)
			force +=  direc* self.wall_kernel(dist) #distance from the outer radius wall
		# radial walls' repulsion
		if self.pos[1] > 1/sqrt(3) * self.pos[0] - 2/sqrt(3) * self.wall_effect_thickness :
			direc = 0.5* ary([1, -sqrt(3)])
			dist = direc.dot(self.pos)
			force += direc* self.wall_kernel(dist)
		elif self.pos[1] < - 1/sqrt(3) * self.pos[0] + 2/sqrt(3) * self.wall_effect_thickness :
			direc = 0.5* ary([1, sqrt(3)])
			dist = direc.dot(self.pos)
			force += direc* self.wall_kernel(dist)
		return force
	
	def walk(self, step):
		self.pos += step

def get_outline(resolution = RESOLUTION):
	"""Get the sextant sheath outline, which is useful for plotting"""
	radians = np.linspace( 0, 2*pi, resolution)
	original_circle = [ [cos(theta), sin(theta)] for theta in radians ]
	outline = circle_to_sextant(original_circle)
	return outline

def easy_logspace(start, stop, num):
	return np.logspace(np.log10(start), np.log10(stop), num)

if __name__=="__main__":
	logger = logging.getLogger(__name__) #created logger with NOSET level of logging, i.e. all messages are recorded.
	# logFormatter = '%(asctime)s - %(levelname)s - %(message)s'
	logger.setLevel(logging.DEBUG)
	logHandler = logging.FileHandler('first_run.log')
	handler = logger.addHandler(logHandler)

	circle = tessellate_circle_properly(417)
	logger.info("Using reference step size={0}, RAD = {1}".format(REF_STEP_SIZE, RAD))
	cable = [] #sorted by [layer=theta][strand_number][xy_coords]
	start_t = time()

	for theta in np.linspace(0, tau, RESOLUTION)[1053:]: #
		new_circle = rotate_list_of_points(circle, theta)
		sextant = ary(circle_to_sextant(new_circle))
		'''
		#plot the initial state
		plt.scatter(sextant[:,0], sextant[:,1])
		plt.title("Intital distribution")
		plt.show()
		'''
		all_points = [ Point(p) for p in sextant ]
		
		step = 0
		force_mag = [quadrature(p.get_force(all_points)) for p in all_points ]
		des = describe(force_mag)
		while des.minmax[1] > F_LIMIT and des.mean > F_LIMIT/2.5: #For F_LIMIT = .5% of radius, this should take a bit more than 200 iterations.
			forces = ary([ p.get_force(all_points) for p in all_points ])
			force_mag = [quadrature(f) for f in forces]
			'''
			plt.hist(force_mag, bins = 200)
			sns.kdeplot(force_mag)
			plt.title("forces magnitudes at step "+str(step))
			plt.show()
			'''
			#get the statistical information about the forces' magnitudes.
			des = describe(force_mag)
			
			#calculate step size to take
			if des.minmax[1] < F_LIMIT*15 and des.mean < F_LIMIT*15/2.5:
				if NEW_KERNEL:
					step_size=0.9
				else:
					step_size = 1.8*REF_STEP_SIZE
			else:
				if NEW_KERNEL:
					step_size=1
				else:
					x = ary([ force_mag[i]/all_points[i].radius_of_effectiveness for i in range(len(force_mag)) ])
					dev_factor = (x)*inv_kernel(1-x) #make sure that it's not deviated by too much.
					step_size = REF_STEP_SIZE*min(dev_factor)

			#print these information for debug purposes.
			logger.debug("Step {0} has a mean force of {1} with bounds of {2} and skewness={3}. Taking a step size of {4}".format( step, des.mean, des.minmax, des.skewness, step_size) )
			#take a walk :)
			for i in range(len(all_points)):
				all_points[i].walk(forces[i]*step_size)
			step += 1
		
		relaxed_pos = ary([p.pos for p in all_points])
		cable.append(relaxed_pos)
		logger.info("Finished angle {0} using {1} steps at t={2}s after the start time".format(theta, step, time()-start_t) )
		if RUNNING_ON_LOCAL:
			#plotting for checking for anomalies.
			plt.scatter(relaxed_pos[:,0], relaxed_pos[:,1] )
			plt.title("Final distribution after relaxing for {} steps".format(NUM_STEPS))
			plt.show()
		else:
			with open ("single_cable_data.txt", "a+") as f:
				f.write(np.array2string(relaxed_pos, separator=','))
				f.write(",\n")
	if not RUNNING_ON_LOCAL:
		np.save('single_cable_data.npy', cable)