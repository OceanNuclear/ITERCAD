from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
from relaxvideo import str2array
from particlerelax import Point
from SimpleMapping import circle_to_sextant, tessellate_circle_properly, rotate_list_of_points
from scipy.stats import describe

START_FROM_PRISTINE = False
RESOLUTION = 200

with open('single_cable_data.txt', 'r') as f:
    data = ('').join(f.readlines()).replace('\n','').replace(']],',']]\n').split('\n')

step_size = len(data)//RESOLUTION
data = ary([ str2array(dat_line[1:-1]) for dat_line in data])[::step_size]

if START_FROM_PRISTINE:
    circle = tessellate_circle_properly(417)
    data = []
    for theta in np.linspace(0, tau, RESOLUTION):
        rotated_circle = rotate_list_of_points(circle, theta)
        transformed_list = circle_to_sextant(rotated_circle)
        data.append(rotated_circle)

def attract_kernel(x):
    return -x

class Slice:
    def __init__(self, list_of_points, ak=attract_kernel):
        self.points = []
        self.ak_raw = ak

        for p in list_of_points:
            #each one should be a Point object
            self.points.append(p)
        self.num_points = len(self.points)
    
    def attract_kernel(self, disp):
        return self.ak_raw(disp) # no scaling needed
    
    def walk(self, final_force_array):
        for i in range(self.num_points):
            self.points[i].walk(final_force_array[i])

    def get_full_internal_forces(self):
        return [ p.get_force_raw(self.points) for p in self.points]

    def get_total_force_raw(self, slice_above, slice_below):
        force_list = self.get_full_internal_forces()
        for i in range(self.num_points):
            disp_above = slice_above.points[i].pos - self.points[i].pos
            disp_below = slice_below.points[i].pos - self.points[i].pos
            force_list[i] = np.append(force_list[i], [self.ak_raw(disp_above), self.ak_raw(disp_below)], axis=0)
        return force_list
    
    def get_total_force(self, slice_above, slice_below):
        force_list = self.get_total_force_raw(slice_above, slice_below)
        averaged_forces = []
        for i in range(self.num_points):
            force = force_list[i]
            averaged_forces.append( [sum(force[:,0])/len(force), sum(force[:,1])/len(force)] )
        return averaged_forces

print("suffer!")
column = []
for cross_section in data:
    column.append( Slice([ Point(pos) for pos in cross_section ]))
for step in range(10):
    column_force = []
    for i in range(len(column)):
        below = i-1
        above = i+1%len(column)
        column_force.append( column[i].get_total_force(column[below], column[above]) )
        print( describe(column_force[i]) )

    for i in range(len(data)):
        column[i].walk( column_force[i] )
    print("\nfinished step", step, "\n")