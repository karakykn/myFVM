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

def read_msh1D_v2(caseName, boundary_conds = [-1, -1]):
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

        areas[i] = np.abs(cells[i,1] - cells[i, 0])

    slopes = np.zeros(cells.shape[0])

    np.savetxt(mesh_path + "points", points)
    np.savetxt(mesh_path + "cells", cells, fmt="%d")
    np.savetxt(mesh_path + "areas", areas)
    np.savetxt(mesh_path + "neighbors", neighbors, fmt="%d")
    np.savetxt(mesh_path + "slopes", slopes)

def read_msh1D(caseName, boundary_conds = [-1, -1]):
    mesh_path = caseName + "/mesh/"
    mesh = meshio.read(caseName + "/mesh/gmsh", file_format="gmsh")

    points = mesh.points
    cells = mesh.cells_dict["line"]
    neighbors = np.zeros_like(cells)
    edges = np.empty((0,3), dtype=int)
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

        edge_cell1 = np.array([cells[i,0], neighbors[i,0], i])  #assembling edge matrix to do flux calculations over unique edges and assign the values
        edge_cell2 = np.array([cells[i,1], i, neighbors[i,1]])
        edges = np.vstack((edges, edge_cell1))
        edges = np.vstack((edges, edge_cell2))

        areas[i] = np.abs(cells[i,1] - cells[i, 0])

    df = pd.DataFrame(edges)
    filtered_df = df.drop_duplicates(subset=0)
    edges = filtered_df.to_numpy()

    np.savetxt(mesh_path + "points", points)
    np.savetxt(mesh_path + "cells", cells)
    np.savetxt(mesh_path + "edges", edges)
    np.savetxt(mesh_path + "areas", areas)
    np.savetxt(mesh_path + "neighbors", neighbors)


caseName = "1dDambreak"
boundaryType = np.loadtxt(caseName + "/mesh/boundaryType")
read_msh1D_v2(caseName, boundaryType)