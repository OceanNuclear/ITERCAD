from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.animation as manimation
from particlerelax import get_outline
from scipy.stats import describe
from repel_attract import get_disp_vec, calc_dist
VIDEO = True
READ_FROM_HUMAN_READABLE_FILE= False
READ_FROM_SIMPL_NPY = True
EVERY_NTH_FRAME=1

target_r_min_to_r_max_ratio = 0.45/1.985 #both radii measurements were recorded in mm.
offset_constant_in_sqrt = 12 * (1 + target_r_min_to_r_max_ratio**2)/(1 - target_r_min_to_r_max_ratio**2)
radius_max = sqrt(offset_constant_in_sqrt+12)/sqrt(pi)
radius_min = sqrt(offset_constant_in_sqrt-12)/sqrt(pi)

def str2array(l):
    '''
    Function for reading data (stored as plain text) into numpy array
    '''

    p=0
    text = ['',]*l.count(']') 
    for i in l: 
        if i==']': 
            p += 1 
        elif i =='[': 
            pass 
        else: 
            text[p] += i 
    return ary([np.fromstring(entry.lstrip(', '), sep=',') for entry in text])

# def reorder_points(slice_below, slice_here):
#     new_order_points = []
#     for i in range(slice_below.num_points):
#         p = slice_below.points[i]
#         .append([ quadrature(disp) for disp in ary(slice_here) - p.pos])

#         new_order_points.append(slice_here.points[j])
#     slice_here.points = new_order_points # need to check if there are duplicates though
#     return slice_here

def check_in_bound(p):
    '''
    Check if the point p is inside the sextant or not, and return a boolean.
    '''
    dist_from_centre = quadrature(p)
    if abs(p[1]/p[0])>tan(pi/6):
        return True
    elif (dist_from_centre>radius_max) or (dist_from_centre<radius_min):
        return True
    else:
        return False

if __name__=='__main__':
    frame_data = []
    if READ_FROM_HUMAN_READABLE_FILE:
        with open('single_cable_data.txt', 'r') as f:
            data = ('').join(f.readlines()).replace('\n','').replace(']],',']]\n').split('\n')
        for i in range(len(data)):
            frame_data.append(str2array(data[i][1:-1]))
    elif READ_FROM_SIMPL_NPY:
        import sys
        if len(sys.argv)>1:
            mask = ary([int(i) for i in sys.argv[1:]], dtype=int)
            frame_data = np.load('repel_attract_skip4.npy')[:, mask, :]
        else:
            frame_data = np.load('repel_attract_skip4.npy')
    else:
        from interframeattract import *
        column = np.load('', allow_pickle=True)
        # coordinates = []
        # for s in column:
        #     coordinates.append([p.pos for p in in s.points])
        for i in range(len(column)):
            frame_data.append( [column[i].points[j].pos for j in range(column[i].num_points)] )

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='linear_result', artist='Matplotlib', comment='100 frames, Triple repel')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    fig, ax = plt.subplots()
    ax.set_aspect(1.0)
    ax.set_xlim((1-sqrt(3)/2)*radius_min, radius_max)
    ax.set_ylim(-1/2*radius_max, 1/2*radius_max)
    ax.set_xticks([])
    ax.set_yticks([])
    outline = ary(get_outline()).T
    ax.plot(outline[0], outline[1])
    xycoods1 = ax.scatter(np.ones(417)[::3], np.zeros(417)[::3])
    xycoods2 = ax.scatter(np.ones(417)[1::3], np.zeros(417)[1::3])
    xycoods3 = ax.scatter(np.ones(417)[2::3], np.zeros(417)[2::3])
    
    distances_of_column = calc_dist(get_disp_vec(frame_data, frame_data))
    min_dists = []
    for dist_of_slice in distances_of_column:
        np.fill_diagonal(dist_of_slice, max(dist_of_slice.flatten()))
        min_dists.append(dist_of_slice.min().copy())

    print(describe(min_dists))

    if VIDEO:
        with writer.saving(fig, "repel_attract_skip4.mp4", 300):
            for i in range(len(frame_data)):
                if i%EVERY_NTH_FRAME==0:
                    xycoods1.set_offsets( frame_data[i][::3] )
                    xycoods2.set_offsets( frame_data[i][1::3] )
                    xycoods3.set_offsets( frame_data[i][2::3] )
                    ax.set_title("Layer "+str(i))
                    writer.grab_frame()
            fig.clf()
    else:
        plt.cla()
        for i in range(len(frame_data)):
            distances = calc_dist(get_disp_vec(frame_data[i]))
            # frame_data[i+1], frame_data[i-1]
            des = describe(distances)
            print(des)
            if sum(n:=[ check_in_bound(p) for p in frame_data[i] ])>0:
                print("Out of bounds points = ")
                print(np.argwhere(n))
            # plt.hist(distances, bins=100)
            # plt.title("spacing for frame"+str(i))
            # plt.show()
            # print(f"for frame {i}, the minimum and maximum spacings are {min(distances)}, {max(distances)}")
