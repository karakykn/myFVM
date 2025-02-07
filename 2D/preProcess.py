import meshio
import numpy as np
import matplotlib.pyplot as plt

"""Define wall (-1),inlet(-2) and outlet(-3) in order in Gmsh
such as:
Physical Curve("wall") = {1, 2, 3, 5, 6, 7};
Physical Curve("inlet") = {8};
Physical Curve("outlet") = {4};
"""

def generate_edge2d(cells, neighbors):
    edges = np.zeros((cells.shape[0]*3,4), dtype=int)
    for i in range(cells.shape[0]):
        for j in range(3):
            edges[3*i+j,0], edges[3*i+j,1] = cells[i,(j+1)%3], cells[i, j]
            edges[3*i+j,2] = i
            edges[3 * i + j, 3] = neighbors[i,j]

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

def triangle_area(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    # Define the vectors AB and AC
    AB = np.array([x2 - x1, y2 - y1, z2 - z1])
    AC = np.array([x3 - x1, y3 - y1, z3 - z1])

    # Compute the cross product of AB and AC
    cross_product = np.cross(AB, AC)

    # The area is half the magnitude of the cross product
    area = 0.5 * np.linalg.norm(cross_product)

    return area

def read_msh(file_path):
    mesh = meshio.read(file_path, file_format="gmsh")

    points = mesh.points
    cells = mesh.cells_dict["triangle"]
    lines = mesh.cells_dict['line']
    neighbors = np.zeros_like(cells)
    areas = np.zeros(cells.shape[0])

    potentialNeighs = []
    for i in range(cells.shape[0]):
        matching_rows = np.any(np.isin(cells, cells[i, :]), axis=1)
        matching_row_indices = np.where(matching_rows)[0]
        matching_row_indices = matching_row_indices[matching_row_indices != i]
        potentialNeighs.append(matching_row_indices)

        areas[i] = triangle_area(points[cells[i, 0], 0], points[cells[i, 0], 1], points[cells[i, 0], 2],
                                 points[cells[i, 1], 0],
                                 points[cells[i, 1], 1],
                                 points[cells[i, 1], 2], points[cells[i, 2], 0], points[cells[i, 2], 1],
                                 points[cells[i, 2], 2])

    for i in range(cells.shape[0]):
        for j in range(3):
            edge_seek = np.array([cells[i, (j+1) % 3], cells[i, j]], dtype=int)
            edge_list = np.zeros((potentialNeighs[i].size * 3, 3), dtype=int)
            for k, potneigh in enumerate(potentialNeighs[i]):
                for l in range(3):
                    edge_list[k * 3 + l, 0] = cells[potneigh, l]
                    edge_list[k * 3 + l, 1] = cells[potneigh, (l+1)%3]
                    edge_list[k * 3 + l, 2] = potneigh
            edge_match = np.all(edge_list[:,:-1] == edge_seek, axis=1)
            if np.any(edge_match):
                indices = np.where(edge_match)[0]
                neighbors[i, j] = edge_list[indices, 2]
            else:
                edge_seek2 = np.array([cells[i, j], cells[i, (j+1) % 3]], dtype=int)
                for pair in [edge_seek, edge_seek2]:
                    idx = np.where(np.all(lines == pair, axis=1))
                neighbors[i, j] = -mesh.cell_data_dict['gmsh:physical']['line'][idx][0]

    slopes = np.zeros((cells.shape[0], 2))
    '''calculate the unit normal of each cell and assign, for now, it is not necessary'''
    return points, cells, neighbors, areas, slopes

def read_msh_v2(caseName):
    mesh_path = caseName + "/mesh/"
    mesh = meshio.read(caseName + "/mesh/gmsh", file_format="gmsh")

    points = mesh.points
    cells = mesh.cells_dict["triangle"].astype(int)
    lines = mesh.cells_dict['line'].astype(int)
    neighbors = np.zeros_like(cells, dtype = int)
    areas = np.zeros(cells.shape[0])

    potentialNeighs = []
    for i in range(cells.shape[0]):
        matching_rows = np.any(np.isin(cells, cells[i, :]), axis=1)
        matching_row_indices = np.where(matching_rows)[0]
        matching_row_indices = matching_row_indices[matching_row_indices != i]
        potentialNeighs.append(matching_row_indices)

        areas[i] = triangle_area(points[cells[i, 0], 0], points[cells[i, 0], 1], points[cells[i, 0], 2],
                                 points[cells[i, 1], 0],
                                 points[cells[i, 1], 1],
                                 points[cells[i, 1], 2], points[cells[i, 2], 0], points[cells[i, 2], 1],
                                 points[cells[i, 2], 2])

    for i in range(cells.shape[0]):
        for j in range(3):
            edge_seek = np.array([cells[i, (j+1) % 3], cells[i, j]], dtype=int)
            edge_list = np.zeros((potentialNeighs[i].size * 3, 3), dtype=int)
            for k, potneigh in enumerate(potentialNeighs[i]):
                for l in range(3):
                    edge_list[k * 3 + l, 0] = cells[potneigh, l]
                    edge_list[k * 3 + l, 1] = cells[potneigh, (l+1)%3]
                    edge_list[k * 3 + l, 2] = potneigh
            edge_match = np.all(edge_list[:,:-1] == edge_seek, axis=1)
            if np.any(edge_match):
                indices = np.where(edge_match)[0]
                neighbors[i, j] = edge_list[indices, 2]
            else:
                edge_seek2 = np.array([cells[i, j], cells[i, (j+1) % 3]], dtype=int)
                for pair in [edge_seek, edge_seek2]:
                    idx = np.where(np.all(lines == pair, axis=1))
                neighbors[i, j] = -mesh.cell_data_dict['gmsh:physical']['line'][idx][0]


    edges = generate_edge2d(cells, neighbors)
    slopes = np.zeros((cells.shape[0], 2))
    """slopes are assigned to zero, edit this line for slopes"""

    np.savetxt(mesh_path + "points", points)
    np.savetxt(mesh_path + "cells", cells, fmt="%d")
    np.savetxt(mesh_path + "areas", areas)
    np.savetxt(mesh_path + "neighbors", neighbors, fmt="%d")
    np.savetxt(mesh_path + "slopes", slopes)
    np.savetxt(mesh_path + "edges", edges, fmt="%d")

def visualize_mesh(caseName):
    mesh_path = caseName + "/mesh/"
    nodes = np.loadtxt(mesh_path + 'points')[:, :-1]
    cells= np.loadtxt(mesh_path + 'cells', dtype=int)
    edges = np.loadtxt(mesh_path + 'edges', dtype=int)
    for k, i in enumerate(edges):
        plt.plot([nodes[i[0]][0],nodes[i[1]][0]],[nodes[i[0]][1],nodes[i[1]][1]], "k")
        plt.text( (nodes[i[0]][0]+nodes[i[1]][0])/2, (nodes[i[0]][1]+nodes[i[1]][1])/2, k, color="r")
    for l in range(cells.shape[0]):
        i = cells[l]
        cellCx, cellCy = (nodes[i[0]][0] + nodes[i[1]][0] + nodes[i[2]][0]) / 3, (nodes[i[0]][1] + nodes[i[1]][1] + nodes[i[2]][1]) / 3
        plt.text(cellCx, cellCy, l, color = "b")
    for i, node in enumerate(nodes):
        plt.text(node[0],node[1],i,color="k")
    plt.title("edges red, cells blue, nodes black")
    plt.savefig(mesh_path+'mesh.pdf')


caseName = "Basicmesh"
# read_msh_v2(caseName)
visualize_mesh(caseName)