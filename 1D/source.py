import meshio
import numpy as np
import matplotlib.pyplot as plt

class Swe1D:
    def __init__(self, caseName, qx_ini, h_ini, g=9.81, n=0.012, length = 1, nodeNo=101, boundaryCond = [-1, -1]):
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

        self.caseName = caseName
        mesh_path = caseName + "/mesh/"
        self.mesh = {
            'nodes': np.loadtxt(mesh_path + 'points')[:,0],
            'cells': np.loadtxt(mesh_path + 'cells', dtype=int),
            'neighbors': np.loadtxt(mesh_path + 'neighbors', dtype=int),
            'lengths': np.loadtxt(mesh_path + 'areas'),
            'slopes': np.loadtxt(mesh_path + 'slopes'),
        }
        edges = []
        for row in self.mesh['cells']:
            edge = [[row[i], row[i + 1]] for i in range(len(row) - 1)] + [[row[-1], row[0]]]
            edges.append(edge)
        self.mesh.update({'edges': edges})
        self.g = g
        self.n = n
        self.length = length
        self.initialize_variables()
        self.initial_conditions(qx_ini, h_ini)

    def initialize_variables(self):
        """Initialize conserved variables and fluxes."""
        num_elements = len(self.mesh['cells'])
        self.U = np.zeros((num_elements, 2))  # Conserved variables [h, h*u, h*v] for 1D # Sources

    def initial_conditions(self, qx_ini,  h_ini):
        for i in range(self.U.shape[0]):
            if self.mesh["nodes"][self.mesh['cells'][i,0]] < self.length / 3:
                self.iniH = h_ini
                self.U[i, 0] = h_ini
        self.iniQx = qx_ini
        self.U[:, 1] = qx_ini


    def compute_flux(self, U_left, U_right, normal):
        """
        Compute the upwind numerical flux at an edge.

        Parameters:
            U_left: ndarray
                Conserved variables on the left side of the edge.
            U_right: ndarray
                Conserved variables on the right side of the edge.
            normal: ndarray
                Unit normal vector of the edge.

        Returns:
            flux: ndarray
                Numerical flux vector.
        """
        h_L, hu_L = U_left
        h_R, hu_R = U_right

        # Calculate velocities
        u_L = hu_L / h_L if h_L > 0 else 0
        u_R = hu_R / h_R if h_R > 0 else 0

        # Compute fluxes
        flux_L = np.array([
            h_L * u_L,
            h_L * u_L * np.abs(u_L) + 0.5 * self.g * h_L ** 2,
        ]) * normal
        flux_R = np.array([
            h_R * u_R,
            h_R * u_R * np.abs(u_R) - 0.5 * self.g * h_R ** 2,
        ])

        # Upwind flux (based on velocity direction)
        if u_L * normal > 0:  # Flow to the right
            return flux_L
        elif u_R * normal < 0:  # Flow to the left
            return flux_R
        else:  # Middle state (Riemann solver)
            return 0.5 * (flux_L + flux_R)

    def compute_source(self, i):
        n = self.n
        g = self.g
        S_0 = self.mesh['slopes'][i]
        h, hu= self.U[i]

        u = hu / h if h > 0 else 0
        tau_b_x = (n ** 2 * u * np.sqrt(u ** 2)) / (h ** (4 / 3) + 1e-8)  # Add small value to prevent division by zero
        source_x = g * h * (S_0 + tau_b_x)
        return 0, source_x

    def update_solution(self, dt):
        """Update the solution using the finite volume method."""
        self.F = np.zeros_like(self.U)  # Fluxes
        self.S = np.zeros_like(self.U)
        edgy = self.mesh['edges']
        for i, edgum in enumerate(edgy):
            for j, tuple in enumerate(edgum):
                neighbor = self.mesh['neighbors'][i,j]
                # Get left and right states
                U_left = self.U[i]
                U_right = np.zeros_like(U_left)
                if neighbor == -1:  # Boundary condition (e.g., wall )
                    U_right[0] = self.U[i][0]
                    U_right[1] = -self.U[i][1]
                elif neighbor == -2: # inlet (fixed h, may switch to hydrograph)
                    U_right[0] = 2*self.iniH - self.U[i][0]
                    U_right[1] = 2 * self.iniQx - self.U[i][1]
                elif neighbor == -3: # outlet
                    U_right[0] = self.U[i][0]
                    U_right[1] = self.U[i][1]
                else:
                    U_right = self.U[neighbor]

                n1, n2 = tuple[0], tuple[1]
                normal = -self.mesh['nodes'][n2] + self.mesh['nodes'][n1]
                normal = normal / np.linalg.norm(normal)  # Normalize

                # Compute flux across the edge
                flux = self.compute_flux(U_left, U_right, normal)
                self.F[i] += flux * normal / self.mesh['lengths'][i]
                if neighbor >= 0:
                    self.F[neighbor] -= flux * normal / self.mesh['lengths'][i]
                    edgy[neighbor] = [sublist for sublist in edgy[neighbor] if sublist[0] != np.int64(tuple[0])]

            self.S[i] = self.compute_source(i)

            # Time integration (Euler method)
            self.U[i] += dt * (-self.F[i] + self.S[i])

    def solve(self, t_max, dt):
        """Run the simulation."""
        t = 0
        while t < t_max:
            self.update_solution(dt)
            t += dt
            print(f"Time: {t:.2f}, Max h: {np.max(self.U[:, 0]):.2f}")

    def iterativeSolve(self, dt, tolerance = 1e-6, print_step = 50):
        """Run the simulation."""
        iter = 0
        residual = 1
        plt.ion()
        iteration = [iter]
        res = [residual]
        oldH = np.zeros(self.U.shape[0])
        while residual > tolerance:
            iter += 1
            oldH[:] = self.U[:,0]
            self.update_solution(dt)
            residual = np.max(np.abs(self.U[:,0] - oldH))
            if iter % print_step == 0:
                print(f"Iteration: {iter}, Residual: {residual}")
                np.savetxt(self.caseName + "/output/h" + str(iter) + ".csv", oldH)
                iteration.append(iter)
                res.append(residual)
                plt.plot(iteration, res)
                plt.show()
                plt.pause(.1)
                plt.cla()
