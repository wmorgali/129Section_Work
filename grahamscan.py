import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randint
from math import atan2

#Ask for a .dat file of points
mesh = input("Enter the path to the .dat file: ")
data = pd.read_csv(mesh, delim_whitespace=True)

x = data['X']
y = data['Y']   

# Plot the data points and save them to mesh.png
plt.scatter(x, y, color='blue', label='Data Points')
plt.title("Data Points from .dat File (Pandas)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('mesh.png', dpi=300)  # Save as a high-quality image (300 DPI)

# Convert the data to a list of points
pts = data[['X', 'Y']].values.tolist()
#Convex Hull
#Creates a scatter plot of the points and the convex hull saved to hull.png
def scatter_plot(coords, convex_hull=None):
    xs, ys = zip(*coords)
    plt.scatter(xs,ys)
    if convex_hull != None:
        for i in range(1,len(convex_hull)+1):
            if i == len(convex_hull): i = 0
            c0 = convex_hull[i-1]
            c1 = convex_hull[i]
            plt.plot((c0[0], c1[0]), (c0[1], c1[1]), 'r')
        plt.savefig('hull.png', dpi=300)  # Save as a high-quality image (300 DPI)

#Polar Angle between 2 points
def polar_angle(p0,p1=None):
    if p1==None: p1 = anchor
    y_span = p0[1]-p1[1]
    x_span = p0[0]-p1[0]
    return atan2(y_span, x_span)

#Distance between 2 points
def distance(p0,p1=None):
    if p1==None: p1 = anchor
    y_span = p0[1]-p1[1]
    x_span = p0[0]-p1[0]
    return y_span**2 + x_span**2

#Determinant of 3 points
def det(p1,p2,p3):
    return   (p2[0]-p1[0])*(p3[1]-p1[1]) \
            -(p2[1]-p1[1])*(p3[0]-p1[0])

#Quicksort
def quicksort(a):
    if len(a)<=1: return a
    smaller,equal,larger = [],[],[]
    piv_ang = polar_angle(a[randint(0,len(a)-1)])
    for pt in a:
        pt_ang = polar_angle(pt)
        if   pt_ang < piv_ang:  smaller.append(pt)
        elif pt_ang == piv_ang: equal.append(pt)
        else:                   larger.append(pt)
    return   quicksort(smaller) \
            +sorted(equal, key=distance) \
            +quicksort(larger)

#Graham Scan function
def graham_scan(points,show_progress=False):
    global anchor
    min_idx = None
    for i,(x,y) in enumerate(points):
        if  min_idx==None or y < points[min_idx][1]:
            min_idx = i
        if  y == points[min_idx][1] and x < points[min_idx][0]:
            min_idx = i
    anchor = points[min_idx]
    sorted_pts = quicksort(points)
    del sorted_pts[sorted_pts.index(anchor)]
    hull = [anchor,sorted_pts[0]]
    for s in sorted_pts[1:]:
        while det(hull[-2],hull[-1],s) <= 0:
            del hull[-1]
            if len(hull)<2: break
        hull.append(s)
        if show_progress: scatter_plot(points,hull)
    return hull

#Runs the function
scatter_plot(pts,graham_scan(pts,False))