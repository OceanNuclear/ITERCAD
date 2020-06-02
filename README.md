The directory is created for simulating the distribution of wire inside a cable. 

# motivation
A superconducting cable of 2500 strands will be manufactured as part of ITER's electromagnet to control the plasma. This calls for a nuclear heating simulation of these components. It is known that the magnetic field produced by these individual wires will affect the path taken by the charged particles, and therefore the path of some ionizing radiation. However, no model has been created that goes down to this level of detail, i.e. has the distribution of individual wires. This repository of code aims to generate such a model.

# Set up
The cable consist of 6 sub-cable twisted around a central (steel) sub-cable; and the sub-cable themselves are also twisted. Therefore we have to simulate ~417 strands of superconducting wires wrapped inside one sub-cable, twisted, and shaped into a truncated sextant (one sixth of a cookie ring), and then duplicate it six times, before pluggin it into CAD. The twisting of the sub-cable shows up as rotation in its cross-section as we move up the cable.

The distribution of wire is found by FEM, by relaxing from an initial distribution of wires. This initial distribution is formed by equipopulating a circle with 417 points, and then "squaring the circle" (mapping the circle into a square), and then perform an area-preserving transformation from the square into the sextant. By rotating this initial distribution of points used for the mapping, we can form a primitive rotation in the distribution of wires in the cross-section as we go up the wire. 

Of course, it would be too convenient if this initial distribution is already optimum (i.e. matches reality). But it is not, since some wires are packed more tightly than others in this distribution. Therefore we spread out these wires in the following simulation using FEM-like procedures, by making them repel near-by wires.

# Method
## Initial method
417 particles confined in the sextant shape are subjected to forces from the walls and from all other points (if they are close enough, i.e. within the effective radius).

They are then displaced by an amount proportional the amount of repulsion it felt. 

The repulsion is then re-calculated in the next iteration.
This was repeated for each slice of the cable as we move up this cable

## Underrelaxation
In some cases if two particles were too close together they will experience extremely high repulsion, and fly out of bounds. To prevent this I have implemented a system that reduces the step size of the particles, reducing the step size until every particle moves less than or equal to a set distance.

## Linking to other layers
In the approach described above, there are no connection between layers, so the relaxation's resulting distribution of particles of two adjacent slices of the sub-cable may be wildly different. Therefore I applied 
- Attraction between corresponding points on adjacent slices (i.e. core of wire A in slice 1 will be attracted to core of wire A in slice 2, etc.)
- Repulsion between neighbouring slices (i.e. core of wire A will be repelled by core of wire B, C, ... in slice 2)

# Attempt
The first attempt used a repulsive kernel that (cot(x)). I thought this was the appropriate approach because repulsive force shoult approach infinity

It also used an extremely slow looping approach, where the force is calculated in a different manner, not utiliing numpy's speed by vectorization. The results were unsatisfying.

# Result
The better version is created in repel_attract.py, using a linear kernel instead (x-1). You can play the video in data/*.mp4 to check out the result of the simultaion
