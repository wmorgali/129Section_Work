"""
from collections import Counter
from scipy.spatial import Delaunay
np.isclose
"""
import numpy as np
import matplotlib.pyplot as plt

# Define the surfaces
def surface1(x, y):
    return 2*x**2 + 2*y**2

def surface2(x, y):
    return 2*np.exp(-x**2 - y**2)

# Generate the surface point cloud
def generate_surface_point_cloud(x_range, y_range, num_points=100):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)

    # Calculate the z values for each surface
    Z1 = surface1(X, Y)
    Z2 = surface2(X, Y)

    # Flatten the arrays to get the point cloud
    points_surface1 = np.vstack((X.flatten(), Y.flatten(), Z1.flatten())).T
    points_surface2 = np.vstack((X.flatten(), Y.flatten(), Z2.flatten())).T

    # Combine the point clouds of the two surfaces
    point_cloud = np.concatenate((points_surface1, points_surface2), axis=0)
    
    return point_cloud

# Generate the point cloud
x_range = (-3, 3)  # Define the x range
y_range = (-3, 3)  # Define the y range
point_cloud = generate_surface_point_cloud(x_range, y_range, num_points=100)

# Optionally plot the surface points for visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
plt.savefig('surface_point_cloud.png')
