import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_dat_file(filename):
    df = pd.read_csv(filename, delim_whitespace=True, skiprows=1, names=["X", "Y"])
    points = df.to_numpy()
    return points

def monotone_chain_convex_hull(points):
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
    
    return np.array(lower[:-1] + upper[:-1])

def plot_convex_hull(points, hull):
    plt.scatter(points[:, 0], points[:, 1], label="Points")
    hull = np.append(hull, [hull[0]], axis=0)  # Close the hull
    plt.plot(hull[:, 0], hull[:, 1], "r-", label="Convex Hull")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Convex Hull using Monotone Chain")
    plt.savefig("monotone_hull.png")
    plt.close()

if __name__ == "__main__":
    filename = "mesh.dat"
    points = read_dat_file(filename)
    convex_hull = monotone_chain_convex_hull(points)
    plot_convex_hull(points, convex_hull)
