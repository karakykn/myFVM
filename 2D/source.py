import meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
import ast
import math

def read_input(caseName):
    with open(caseName + '/input', "r") as file:
        lines = file.readlines()
    startTime = int(lines[1][:-1])
    n = float(lines[4][:-1])
    CFL = float(lines[7][:-1])
    simTime = float(lines[10][:-1])
    printStep = float(lines[13][:-1])
    inih = float(lines[17][:-1])
    iniu = float(lines[21][:-1])
    iniv = float(lines[25][:-1])
    boundh = ast.literal_eval(lines[16])
    boundu = ast.literal_eval(lines[20])
    boundv = ast.literal_eval(lines[24])
    interactiveplot = float(lines[28][:-1])
    visualize = int(lines[31])
    inlet = ast.literal_eval(lines[34])
    file.close()
    return [startTime,n,CFL,simTime, printStep,boundh,inih,boundu,iniu,boundv,iniv,interactiveplot, inlet, visualize]

def generate_edge2d(cells, neighbors):
    edges = np.zeros((cells.shape[0]*3,4), dtype=int)
    for i in range(cells.shape[0]):
        print(f'Generating edges: {i+1:d}/{cells.shape[0]:d}')
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

def read_msh(caseName):
    print('Preparing mesh information for the solver...')
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
        print(f'Generating neighbors: {i+1:d}/{cells.shape[0]:d}')
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
    N = cells.shape[0]
    fontS = 50 / N
    for k, i in enumerate(edges):
        if i[3] == -1:
            plt.plot([nodes[i[0]][0], nodes[i[1]][0]], [nodes[i[0]][1], nodes[i[1]][1]], "r")
        else:
            plt.plot([nodes[i[0]][0],nodes[i[1]][0]],[nodes[i[0]][1],nodes[i[1]][1]], "k")
        plt.text( (nodes[i[0]][0]+nodes[i[1]][0])/2, (nodes[i[0]][1]+nodes[i[1]][1])/2, k, color="r", fontsize=fontS)
    for l in range(cells.shape[0]):
        i = cells[l]
        cellCx, cellCy = (nodes[i[0]][0] + nodes[i[1]][0] + nodes[i[2]][0]) / 3, (nodes[i[0]][1] + nodes[i[1]][1] + nodes[i[2]][1]) / 3
        plt.text(cellCx, cellCy, l, color = "b", fontsize=fontS)
    for i, node in enumerate(nodes):
        plt.text(node[0],node[1],i,color="k", fontsize=fontS)
    plt.title("edges red, cells blue, nodes black")
    plt.savefig(mesh_path+'mesh.pdf')

def initial_h(caseName, startTime, bound_h = [[0,1],[0,1]], h_assign = 1):
    print('Creating initial h file...')
    mesh_path = caseName + "/mesh/"
    mesh = {
        'points': np.loadtxt(mesh_path + 'points')[:, :-1],
        'cells': np.loadtxt(mesh_path + 'cells', dtype=int),
        'neighbors': np.loadtxt(mesh_path + 'neighbors', dtype=int),
        'lengths': np.loadtxt(mesh_path + 'areas'),
        'slopes': np.loadtxt(mesh_path + 'slopes'),
        'edges': np.loadtxt(mesh_path + 'edges', dtype=int)
    }

    cellN = mesh['cells'].shape[0]
    h = np.zeros(cellN)

    x1h,x2h, y1h, y2h = bound_h[0][0], bound_h[0][1], bound_h[1][0], bound_h[1][1]
    for i, cell in enumerate(mesh['cells']):
        cellCx = (mesh['points'][cell[0],0] + mesh['points'][cell[1],0]) / 2
        cellCy = (mesh['points'][cell[0],1] + mesh['points'][cell[1],1]) / 2
        if cellCx <= x2h and cellCx >= x1h and cellCy <= y2h and cellCy >= y1h:
            h[i] = h_assign

    time_folder = f"{caseName}/run/{startTime}"
    os.makedirs(time_folder, exist_ok=True)
    np.savetxt(f"{time_folder}/h.csv", h)

def initial_u(caseName, startTime, bound_u=[[0,1],[0,1]], u_assign=0):
    print('Creating initial u file...')
    mesh_path = caseName + "/mesh/"
    mesh = {
        'points': np.loadtxt(mesh_path + 'points')[:, :-1],
        'cells': np.loadtxt(mesh_path + 'cells', dtype=int),
        'neighbors': np.loadtxt(mesh_path + 'neighbors', dtype=int),
        'lengths': np.loadtxt(mesh_path + 'areas'),
        'slopes': np.loadtxt(mesh_path + 'slopes'),
        'edges': np.loadtxt(mesh_path + 'edges', dtype=int)
        }

    cellN = mesh['cells'].shape[0]
    u = np.zeros(cellN)

    x1u,x2u, y1u, y2u = bound_u[0][0], bound_u[0][1], bound_u[1][0], bound_u[1][1]
    for i, cell in enumerate(mesh['cells']):
        cellCx = (mesh['points'][cell[0],0] + mesh['points'][cell[1],0]) / 2
        cellCy = (mesh['points'][cell[0],1] + mesh['points'][cell[1],1]) / 2
        if cellCx <= x2u and cellCx >= x1u and cellCy <= y2u and cellCy >= y1u:
            u[i] = u_assign

    time_folder = f"{caseName}/run/{startTime}"
    os.makedirs(time_folder, exist_ok=True)
    np.savetxt(f"{time_folder}/u.csv", u)

def initial_v(caseName, startTime, bound_v=[[0,1],[0,1]], v_assign=0):
    print('Creating initial v file...')
    mesh_path = caseName + "/mesh/"
    mesh = {
        'points': np.loadtxt(mesh_path + 'points')[:, :-1],
        'cells': np.loadtxt(mesh_path + 'cells', dtype=int),
        'neighbors': np.loadtxt(mesh_path + 'neighbors', dtype=int),
        'lengths': np.loadtxt(mesh_path + 'areas'),
        'slopes': np.loadtxt(mesh_path + 'slopes'),
        'edges': np.loadtxt(mesh_path + 'edges', dtype=int)
        }

    cellN = mesh['cells'].shape[0]
    v = np.zeros(cellN)

    x1v, x2v, y1v, y2v = bound_v[0][0], bound_v[0][1], bound_v[1][0], bound_v[1][1]
    for i, cell in enumerate(mesh['cells']):
        cellCx = (mesh['points'][cell[0],0] + mesh['points'][cell[1],0]) / 2
        cellCy = (mesh['points'][cell[0],1] + mesh['points'][cell[1],1]) / 2
        if cellCx <= x2v and cellCx >= x1v and cellCy <= y2v and cellCy >= y1v:
            v[i] = v_assign

    time_folder = f"{caseName}/run/{startTime}"
    os.makedirs(time_folder, exist_ok=True)
    np.savetxt(f"{time_folder}/v.csv", v)

class Swe2D:
    def __init__(self, caseName, startTime=0, g=9.81, n=0.012, inlet=[1,0,0]):
        """
        Initialize the solver for Saint-Venant equations.

        Parameters:
            mesh: dict
                Contains mesh information (nodes, cells, neighbors, areas, etc.).
                `mesh` should include:
                - 'nodes': Array of node coordinates [[x1, y1], [x2, y2], ...]
                - 'cells': Array of triangles defined by node indices [[n1, n2, n3], ...]
                - 'neighbors': List of neighboring cells for each triangle
                - 'areas': Array of triangle areas
            mode: str
                Either "2D" or "1D" mode. Determines how the solver operates.
            g: float
                Gravitational acceleration.
        """
        mesh_path = caseName + "/mesh/"
        self.mesh = {
            'nodes': np.loadtxt(mesh_path + 'points')[:,:-1],
            'cells': np.loadtxt(mesh_path + 'cells', dtype=int),
            'neighbors': np.loadtxt(mesh_path + 'neighbors', dtype=int),
            'areas': np.loadtxt(mesh_path + 'areas'),
            'slopes': np.loadtxt(mesh_path + 'slopes'),
            'edges': np.loadtxt(mesh_path + 'edges', dtype=int)
        }
        self.g = g
        self.n = n
        self.start = startTime
        self.inlet = inlet
        self.caseName = caseName
        cellN = self.mesh['cells'].shape[0]
        lengths = np.zeros((cellN, 2))
        for i, cell in enumerate(self.mesh['cells']):
            lengths[i,0] = np.max(self.mesh['nodes'][cell,0]) - np.min(self.mesh['nodes'][cell,0])
            lengths[i, 1] = np.max(self.mesh['nodes'][cell, 1]) - np.min(self.mesh['nodes'][cell, 1])
        self.mesh.update({'lengths': lengths})
        self.length_x = np.max(self.mesh['nodes'][:,0]) - np.min(self.mesh['nodes'][:,0])
        self.length_y = np.max(self.mesh['nodes'][:,1]) - np.min(self.mesh['nodes'][:,1])
        self.initialize_variables()
        self.initial_conditions()

    def initialize_variables(self):
        """Initialize conserved variables and fluxes."""
        num_elements = len(self.mesh['cells'])
        self.U = np.zeros((num_elements, 3))  # Conserved variables [h, h*u, h*v] for 1D

    def initial_conditions(self):
        self.U[:,0] = np.loadtxt(self.caseName+"/run/" + str(self.start)+'/h.csv')
        self.U[:,1] = np.loadtxt(self.caseName+"/run/" + str(self.start) + '/u.csv') * self.U[:,0]
        self.U[:, 2] = np.loadtxt(self.caseName + "/run/" + str(self.start) + '/v.csv') * self.U[:, 0]

    def update_boundaries(self):
        self.iniH = self.inlet[0]
        self.iniQx = self.inlet[1] * self.iniH
        self.iniQy = self.inlet[2] * self.iniH

    def hll_flux(self, left_u, right_u, normal):
        hL, uL, vL = left_u
        hR, uR, vR = right_u

        cL = np.sqrt(self.g * hL)
        cR = np.sqrt(self.g * hR)
        unL = uL * normal[0] + vL * normal[1]
        unR = uR * normal[0] + vR * normal[1]
        SL = min(unL - cL, unR - cR)
        SR = max(unL + cL, unR + cR)

        fluxL = np.array([hL * unL,
                          hL * ((uL ** 2 + self.g * hL) * normal[0] + uL * vL * normal[1]),
                          hL * (uL * vL * normal[0] + (vL ** 2 + self.g * hL ) * normal[1])])
        fluxR = np.array([hR * unR,
                          hR * ((uR ** 2 + self.g * hR) * normal[0] + uR * vR * normal[1]),
                          hR * (uR * vR * normal[0] + (vR ** 2 + self.g * hR ) * normal[1])])

        if SL > 0:
            return fluxL
        elif SR < 0:
            return fluxR
        else:
            return (SR * fluxL - SL * fluxR + SR * SL * (right_u - left_u)) / (SR - SL + 1e-8)

    def compute_source(self, i):
        n = self.n
        g = self.g
        S_0x, S_0y = self.mesh['slopes'][i, 0], self.mesh['slopes'][i, 1]
        h, hu, hv = self.U[i]
        u = hu / h if h > 0 else 0
        v = hv / h if h > 0 else 0

        vel_mag = np.sqrt(u**2 + v**2)
        S_fx = self.n ** 2 * u * vel_mag / h ** (4 / 3) if h > 1e-3 else 0
        S_fy = self.n ** 2 * v * vel_mag / h ** (4 / 3) if h > 1e-3 else 0
        source_x = (S_0x - S_fx) * self.g * h
        source_y = (S_0y - S_fy) * self.g * h
        return 0, source_x, source_y

    def update_solution(self, dt):
        """Update the solution using the finite volume method."""
        self.F = np.zeros_like(self.U)  # Fluxes
        self.G = np.zeros_like(self.U)
        self.S = np.zeros_like(self.U)  # Sources
        edges = self.mesh['edges']
        for edge in edges:
            n2, n1 = edge[0], edge[1]
            cell = edge[2]
            neighbor = edge[3]
            U_left = self.U[cell]
            U_right = np.zeros_like(U_left)

            edge_vector = self.mesh['nodes'][n2] - self.mesh['nodes'][n1]
            normal = np.array([edge_vector[1], -edge_vector[0]])
            # normal = normal / np.linalg.norm(normal)
            phi = np.arctan2(normal[1] , normal[0])

            if neighbor == -1:  # Boundary condition (e.g., wall )
                U_right[0] = self.U[cell][0]
                U_right[1] = self.U[cell][1] * (np.sin(phi)**2 - np.cos(phi)**2) - 2 * self.U[cell][2] * np.sin(phi) * np.cos(phi)
                U_right[2] = - 2 * self.U[cell][1] * np.sin(phi) * np.cos(phi) + self.U[cell][2] * (np.cos(phi)**2 - np.sin(phi)**2)
            elif neighbor == -2:  # inlet (fixed h, may switch to hydrograph)
                U_right[0] = 2 * self.iniH - self.U[cell][0]
                U_right[1] = 2 * self.iniQx - self.U[cell][1]
                U_right[2] = 2 * self.iniQy - self.U[cell][2]
            elif neighbor == -3:  # outlet
                U_right[0] = self.U[cell][0]
                U_right[1] = self.U[cell][1]
                U_right[2] = self.U[cell][2]
            else:
                U_right = self.U[neighbor]

            flux = self.hll_flux(U_left, U_right, normal)
            self.F[cell] += flux / self.mesh['areas'][cell]
            if neighbor >= 0:
                self.F[neighbor] -= flux / self.mesh['areas'][neighbor]

        for i in range(self.mesh["cells"].shape[0]):
            self.S[i] = self.compute_source(i)
            self.U[i] += dt * (-self.F[i] + self.S[i])

    def iterativeSolve(self, CFL, simTime = 10, print_step = 200):
        """Run the simulation."""
        iter = 0
        time = self.start
        residual = 1
        plt.ion()
        iteration = [iter]
        res = [residual]
        oldU = np.zeros_like(self.U)
        while time < simTime:
            h = self.U[:, 0]
            u = self.U[:, 1] / (self.U[:, 0] + 1e-8)
            v = self.U[:, 2] / (self.U[:, 0] + 1e-8)
            wave_speed = (self.g * h) ** 0.5
            dt_array = CFL * ( (np.abs(u[:]) + wave_speed + 1e-8) / self.mesh['lengths'][:,0] + (np.abs(v[:]) + wave_speed + 1e-8) / self.mesh['lengths'][:,1] ) ** (-1)
            dt = np.nanmin(dt_array)
            iter += 1
            time += dt
            oldU[:,:] = self.U[:,:]
            self.update_boundaries()
            self.update_solution(dt)
            residual = np.max(np.abs(self.U[:] - oldU[:]))
            if iter % print_step == 0:
                print(f"Time: {time}, Residual: {residual}")
                time_folder = f"{self.caseName}/run/{time:.4f}"
                os.makedirs(time_folder, exist_ok=True)
                np.savetxt(f"{time_folder}/h.csv", self.U[:, 0])
                np.savetxt(f"{time_folder}/u.csv", self.U[:, 1] / (self.U[:,0] + 1e-8))
                np.savetxt(f"{time_folder}/v.csv", self.U[:, 2] / (self.U[:, 0] + 1e-8))
                iteration.append(iter)
                res.append(residual)
                plt.semilogy(iteration, res)
                plt.xlabel("Iteration no")
                plt.ylabel("Residual")
                plt.show()
                plt.pause(.1)
                plt.cla()
        time_folder = f"{self.caseName}/run/{time:.4f}"
        os.makedirs(time_folder, exist_ok=True)
        np.savetxt(f"{time_folder}/h.csv", self.U[:, 0])
        np.savetxt(f"{time_folder}/u.csv", self.U[:, 1] / (self.U[:,0] + 1e-8))
        np.savetxt(f"{time_folder}/v.csv", self.U[:, 2] / (self.U[:, 0] + 1e-8))

    def plot_2d(self, interactive=0):  # set the max min of the contours for h, u, v
        caseName = self.caseName
        run_path = caseName + '/run/'

        mesh = self.mesh
        time_dirs = sorted([d for d in os.listdir(run_path) if d.replace('.', '', 1).isdigit()], key=float)
        if interactive==0:
            time_dirs = [time_dirs[-1]]

        # Extract node coordinates (x, y) from the mesh
        x_nodes = np.array([node[0] for node in mesh['nodes']])
        y_nodes = np.array([node[1] for node in mesh['nodes']])

        # Initialize figure for plotting
        fig, (ax_h, ax_u, ax_v) = plt.subplots(3, 1, figsize=(6, 8))
        triang = tri.Triangulation(x_nodes, y_nodes, mesh['cells'])

        # Turn on interactive mode
        plt.ion()

        hmin, hmax = 1e8, -1e8
        umin, umax = 1e8, -1e8
        vmin, vmax = 1e8, -1e8

        for time_dir in time_dirs:
            h_file = os.path.join(run_path, time_dir, 'h.csv')  # Water level file (h)
            u_file = os.path.join(run_path, time_dir, 'u.csv')  # Velocity in x-direction (u)
            v_file = os.path.join(run_path, time_dir, 'v.csv')  # Velocity in y-direction (v)

            h_data = np.loadtxt(h_file)
            u_data = np.loadtxt(u_file)
            v_data = np.loadtxt(v_file)
            hmin = np.minimum(hmin, np.min(h_data))
            hmax = np.maximum(hmax, np.max(h_data))
            umin = np.minimum(umin, np.min(u_data))
            umax = np.maximum(umax, np.max(u_data))
            vmin = np.minimum(vmin, np.min(v_data))
            vmax = np.maximum(vmax, np.max(v_data))

        # Loop over each time step in the time_dirs
        for time_dir in time_dirs:
            # Read the time file
            time = float(time_dir)  # Assuming last value is the current time step

            # Read the h, u, v data files for the current time step
            h_file = os.path.join(run_path, time_dir, 'h.csv')  # Water level file (h)
            u_file = os.path.join(run_path, time_dir, 'u.csv')  # Velocity in x-direction (u)
            v_file = os.path.join(run_path, time_dir, 'v.csv')  # Velocity in y-direction (v)

            h_data = np.loadtxt(h_file)
            u_data = np.loadtxt(u_file)
            v_data = np.loadtxt(v_file)

            contour_h = ax_h.tripcolor(triang, facecolors=h_data, cmap='plasma', vmin=hmin, vmax=hmax)
            ax_h.set_title('h values')

            # Plot for u_values on ax_u
            contour_u = ax_u.tripcolor(triang, facecolors=u_data, cmap='plasma', vmin=umin, vmax=umax)
            ax_u.set_title('u values')

            # Plot for v_values on ax_v
            contour_v = ax_v.tripcolor(triang, facecolors=v_data, cmap='plasma', vmin=vmin, vmax=vmax)
            ax_v.set_title('v values')

            fig.suptitle(f'Time: {time:.2f} seconds', fontsize=16)

            plt.tight_layout()
            plt.show()
            plt.pause(.2)
            plt.cla()

        plt.ioff()
        contour_h = ax_h.tripcolor(triang, facecolors=h_data, cmap='plasma', vmin=hmin, vmax=hmax)
        ax_h.set_title('h values')

        # Plot for u_values on ax_u
        contour_u = ax_u.tripcolor(triang, facecolors=u_data, cmap='plasma', vmin=umin, vmax=umax)
        ax_u.set_title('u values')

        # Plot for v_values on ax_v
        contour_v = ax_v.tripcolor(triang, facecolors=v_data, cmap='plasma', vmin=vmin, vmax=vmax)
        ax_v.set_title('v values')
        plt.tight_layout()
        fig.colorbar(contour_h, ax=ax_h, label='h values')
        fig.colorbar(contour_u, ax=ax_u, label='u values')
        fig.colorbar(contour_v, ax=ax_v, label='v values')
        plt.show()