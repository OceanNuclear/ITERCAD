from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
from particlerelax import get_outline
from repel_attract import get_disp_vec, calc_dist
from SimpleMapping import rotate
from matplotlib.patches import Circle
target_r_min_to_r_max_ratio = 0.45/1.985 #both radii measurements were recorded in mm.
offset_constant_in_sqrt = 12 * (1 + target_r_min_to_r_max_ratio**2)/(1 - target_r_min_to_r_max_ratio**2)
radius_max = sqrt(offset_constant_in_sqrt+12)/sqrt(pi)
radius_min = sqrt(offset_constant_in_sqrt-12)/sqrt(pi)

if __name__=='__main__':
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='linear_result', artist='Matplotlib', comment='100 frames, Triple repel')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    fig, ax = plt.subplots()
    ax.set_aspect(1.0)
    ax.set_xlim(-radius_max*1.05, radius_max*1.05)
    ax.set_ylim(-radius_max*1.05, radius_max*1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    circle = Circle((0, 0), radius_min, facecolor='grey', edgecolor="none", linewidth=3, alpha=0.5)
    ax.add_patch(circle)
    ax.text(0,0, "central\ncolumn", va="center", ha="center")
    outline = []
    for sex in range(6):
        outline_temp = rotate(ary(get_outline()).T, tau/6*sex)
        outline.append( ax.plot(outline_temp[0], outline_temp[1])[0] )
        # sheath of the bundles
    demo_point = (radius_min+radius_max)/2 *1.3, 0
    demo_center = ary([(radius_min+radius_max)/2, 0])
    demo_scatter = ax.scatter( *demo_point )
    num_frames = 80
    angle_incr = tau/num_frames

    with writer.saving(fig, "sketch.mp4", 300):
        for step in range(num_frames):
            for sex in range(6):
                outline[sex].set_data( rotate(ary(outline[sex].get_data()), angle_incr) )
            demo_point = demo_scatter.get_offsets()[0]
            demo_point = rotate(demo_point - demo_center, angle_incr) + demo_center # rotate around the demo_center
            # remember to rotate the entire frame of reference
            demo_point = rotate(demo_point, angle_incr)
            demo_center = rotate(demo_center, angle_incr)
            demo_scatter.set_offsets(demo_point)
            writer.grab_frame()
        fig.clf()
