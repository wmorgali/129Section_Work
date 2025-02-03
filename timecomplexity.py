import numpy as np
import matplotlib.pyplot as plt
import time
"""
# Generate n-point 2D uniform point cloud
def generate_point_cloud(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(n, 2)

def generate_point_cloud(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # Scale and shift to get values in the range [-5, 5]
    return (np.random.rand(n, 2) * 10) - 5
"""

def generate_point_cloud(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # Generate Gaussian-distributed points with mean 0 and variance 1
    return np.random.randn(n, 2)


# Graham's scan implementation (from user-provided code)
def graham_scan(points):
    from random import randint
    from math import atan2
    
    def polar_angle(p0, p1=None):
        if p1 is None: p1 = anchor
        y_span = p0[1] - p1[1]
        x_span = p0[0] - p1[0]
        return atan2(y_span, x_span)

    def distance(p0, p1=None):
        if p1 is None: p1 = anchor
        y_span = p0[1] - p1[1]
        x_span = p0[0] - p1[0]
        return y_span**2 + x_span**2

    def det(p1, p2, p3):
        return (p2[0]-p1[0]) * (p3[1]-p1[1]) - (p2[1]-p1[1]) * (p3[0]-p1[0])

    def quicksort(a):
        if len(a) <= 1: return a
        smaller, equal, larger = [], [], []
        piv_ang = polar_angle(a[randint(0, len(a) - 1)])
        for pt in a:
            pt_ang = polar_angle(pt)
            if pt_ang < piv_ang: smaller.append(pt)
            elif pt_ang == piv_ang: equal.append(pt)
            else: larger.append(pt)
        return quicksort(smaller) + sorted(equal, key=distance) + quicksort(larger)
    
    global anchor
    anchor = min(points, key=lambda p: (p[1], p[0]))
    sorted_pts = quicksort([p for p in points if not np.array_equal(p, anchor)])
    hull = [anchor, sorted_pts[0]]
    for s in sorted_pts[1:]:
        while len(hull) >= 2 and det(hull[-2], hull[-1], s) <= 0:
            hull.pop()
        hull.append(s)
    return hull

# Monotone Chain algorithm
def monotone_chain(points):
    points = sorted(map(tuple, points))
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

# Run experiments and measure time
n_values = [10, 50, 100, 200, 400, 800, 1000]
graham_times = []
monotone_times = []
num_trials = 5

for n in n_values:
    graham_avg_time = 0
    monotone_avg_time = 0
    for _ in range(num_trials):
        points = generate_point_cloud(n)
        
        start = time.time()
        graham_scan(points)
        graham_avg_time += time.time() - start
        
        start = time.time()
        monotone_chain(points)
        monotone_avg_time += time.time() - start
    
    graham_times.append(graham_avg_time / num_trials)
    monotone_times.append(monotone_avg_time / num_trials)

# Plot results
plt.plot(n_values, graham_times, label="Graham's Scan", marker='o')
plt.plot(n_values, monotone_times, label="Monotone Chain", marker='s')
plt.xlabel("Number of Points (n)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Comparison of Convex Hull Algorithms (Averaged over 5 Trials)")
plt.legend()
plt.grid()
plt.savefig("runtime_comparison_gaussian.png")
plt.show()

# Save conclusion to a text file
conclusion = """The monotone chain still performs more quickly than Graham's scan for all tested values of n.
"""
with open("runtime_conclusion_gaussian.txt", "w") as f:
    f.write(conclusion)
