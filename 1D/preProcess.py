import matplotlib.pyplot as plt
import meshio
import numpy as np
import pandas as pd

"""Define wall (-1),inlet(-2) and outlet(-3) in order in Gmsh
such as:
Physical Curve("wall") = {1, 2, 3, 5, 6, 7};
Physical Curve("inlet") = {8};
Physical Curve("outlet") = {4};
"""

def generate_edge1d(cells, neighbors):
    edges = np.zeros((cells.shape[0]*2,4), dtype=int)
    for i in range(cells.shape[0]):
        for j in range(2):
            edges[2*i+j,0], edges[2*i+j,1] = cells[i,(j+1)%2], cells[i, j]
            edges[2*i+j,2] = i
            edges[2 * i + j, 3] = neighbors[i,(j+1)%2]

    edges_list = edges.tolist()  # Convert to list for easier manipulation
    i = 0
    while i < len(edges_list):
        tup = edges_list[i][2:]
        revTup = [tup[1], tup[0]]

        for j in range(i + 1, len(edges_list)):
            if edges_list[j][2:] == revTup:
                edges_list.pop(j)  # Remove reverse pair
                break

        i += 1  # Only increment if no deletion happened
    return np.array(edges_list)

def read_msh1D(caseName, boundary_conds = [-1, -1]):
    mesh_path = caseName + "/mesh/"
    mesh = meshio.read(caseName + "/mesh/gmsh", file_format="gmsh")

    points = mesh.points
    cells = mesh.cells_dict["line"]
    neighbors = np.zeros_like(cells)
    areas = np.zeros(cells.shape[0])

    for i in range(cells.shape[0]):

        for point in cells[i]:
            matching = np.isin(cells, point)
            row, col = np.where(matching)
            if np.any(row != i):  # Check if there is any value other than i in row
                ind = np.where(row == i)
                row = row[row != i]
                neighbors[i, col[ind]] = row[0]
            else:
                neighbors[i, col] = boundary_conds[col[0]]

        areas[i] = np.abs(points[cells[i,1], 0] - points[cells[i, 0], 0])

    slopes = np.zeros(cells.shape[0])

    edges = generate_edge1d(cells, neighbors)

    np.savetxt(mesh_path + "points", points)
    np.savetxt(mesh_path + "cells", cells, fmt="%d")
    np.savetxt(mesh_path + "areas", areas)
    np.savetxt(mesh_path + "neighbors", neighbors, fmt="%d")
    np.savetxt(mesh_path + "slopes", slopes)
    np.savetxt(mesh_path + "edges", edges, fmt='%d')


caseName = "1dDambreakBenchmark"
boundaryType = np.loadtxt(caseName + "/mesh/boundaryType")
read_msh1D(caseName, boundaryType)