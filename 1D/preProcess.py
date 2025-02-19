import matplotlib.pyplot as plt
import meshio
import numpy as np
import os
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

def initial_h(caseName, startTime, bound_h = [0,1], h_assign = 1):
    mesh_path = caseName + "/mesh/"
    mesh = {
        'points': np.loadtxt(mesh_path + 'points')[:, 0],
        'cells': np.loadtxt(mesh_path + 'cells', dtype=int),
        'neighbors': np.loadtxt(mesh_path + 'neighbors', dtype=int),
        'lengths': np.loadtxt(mesh_path + 'areas'),
        'slopes': np.loadtxt(mesh_path + 'slopes'),
        'edges': np.loadtxt(mesh_path + 'edges', dtype=int)
    }

    cellN = mesh['cells'].shape[0]
    h = np.zeros(cellN)

    x1h,x2h = bound_h[0], bound_h[1]
    for i, cell in enumerate(mesh['cells']):
        cellC = (mesh['points'][cell[0]] + mesh['points'][cell[1]]) / 2
        if cellC <= x2h and cellC >= x1h:
            h[i] = h_assign

    time_folder = f"{caseName}/run/{startTime}"
    os.makedirs(time_folder, exist_ok=True)
    np.savetxt(f"{time_folder}/h.csv", h)

def initial_u(caseName, startTime, bound_u=[0, 1], u_assign=1):
    mesh_path = caseName + "/mesh/"
    mesh = {
        'points': np.loadtxt(mesh_path + 'points')[:, 0],
        'cells': np.loadtxt(mesh_path + 'cells', dtype=int),
        'neighbors': np.loadtxt(mesh_path + 'neighbors', dtype=int),
        'lengths': np.loadtxt(mesh_path + 'areas'),
        'slopes': np.loadtxt(mesh_path + 'slopes'),
        'edges': np.loadtxt(mesh_path + 'edges', dtype=int)
        }

    cellN = mesh['cells'].shape[0]
    u = np.zeros(cellN)

    x1u, x2u = bound_u[0], bound_u[1]
    for i, cell in enumerate(mesh['cells']):
        cellC = (mesh['points'][cell[0]] + mesh['points'][cell[1]]) / 2
        if cellC <= x2u and cellC >= x1u:
            u[i] = u_assign

    time_folder = f"{caseName}/run/{startTime}"
    os.makedirs(time_folder, exist_ok=True)
    np.savetxt(f"{time_folder}/u.csv", u)



caseName = "1dDambreakBenchmark"
startTime = 0

'''adapting mesh for the solver'''
boundaryType = np.loadtxt(caseName + "/mesh/boundaryType")
read_msh1D(caseName, boundaryType)

'''setting initials'''
initial_h(caseName, startTime, [0, 1000], 10)
initial_h(caseName, startTime, [1000+1e8, 2000], 10 * .05)
initial_u(caseName, startTime, [0, 2000], 0)