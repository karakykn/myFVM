import os
import numpy as np
import matplotlib.pyplot as plt

# Define case name and paths
caseName = '1dDambreakBenchmark'
mesh_path = caseName + '/mesh/'
output_path = caseName + '/output/'
h_ini = 0.2

# Load mesh data
mesh = {
    'nodes': np.loadtxt(mesh_path + 'points')[:, 0],
    'cells': np.loadtxt(mesh_path + 'cells', dtype=int),
    'neighbors': np.loadtxt(mesh_path + 'neighbors', dtype=int),
    'lengths': np.loadtxt(mesh_path + 'areas'),
    'slopes': np.loadtxt(mesh_path + 'slopes'),
    'edges': np.loadtxt(mesh_path + 'edges', dtype=int)
}

# Get a sorted list of time directories in output_path
time_dirs = sorted([d for d in os.listdir(output_path) if d.isdigit()], key=int)

# Set up the plot
plt.ion()
fig, ax = plt.subplots()

for time in time_dirs:
    h_file = os.path.join(output_path, time, "h.csv")
    try:
        h_values = np.loadtxt(h_file)
    except OSError:
        continue  # Skip if file is missing

    ax.clear()
    for i, cell in enumerate(mesh['cells']):
        ax.plot([mesh['nodes'][cell[0]], mesh['nodes'][cell[1]]], [h_values[i], h_values[i]], "b")

    ax.set_title(f"Time Step: {time}")
    ax.set_ylim(0, h_ini * 1.2)
    ax.set_xlabel("Position")
    ax.set_ylabel("Water Level")
    plt.pause(0.1)

plt.ioff()
plt.show()
