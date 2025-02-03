import meshio
import numpy as np
import matplotlib.pyplot as plt

class Swe2D:
    def __init__(self, caseName, qx_ini, qy_ini, h_ini, g=9.81, n=0.012):
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
            'nodes': np.loadtxt(mesh_path + 'points')[:,-1],
            'cells': np.loadtxt(mesh_path + 'cells', dtype=int),
            'neighbors': np.loadtxt(mesh_path + 'neighbors', dtype=int),
            'areas': np.loadtxt(mesh_path + 'areas'),
            'slopes': np.loadtxt(mesh_path + 'slopes')
        }
        edges = []
        for row in self.mesh['cells']:
            edge = [[row[i], row[i + 1]] for i in range(len(row) - 1)] + [[row[-1], row[0]]]
            edges.append(edge)
        self.mesh.update({'edges': edges})
        self.g = g
        self.n = n
        self.initialize_variables()
        self.initial_conditions(qx_ini, qy_ini, h_ini)

    def initialize_variables(self):
        """Initialize conserved variables and fluxes."""
        num_elements = len(self.mesh['cells'])
        self.U = np.zeros((num_elements, 3))  # Conserved variables [h, h*u, h*v] for 1D
        self.F = np.zeros_like(self.U)  # Fluxes
        self.S = np.zeros_like(self.U)  # Sources
        self.cellH = np.zeros(num_elements)

    def initial_conditions(self, qx_ini, qy_ini, h_ini):
        self.iniH = h_ini
        self.iniQx = qx_ini
        self.iniQy = qy_ini
        self.U[:, 0] = h_ini
        self.U[:, 1] = qx_ini
        self.U[:, 2] = qy_ini


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
        h_L, hu_L, hv_L = U_left
        h_R, hu_R, hv_R = U_right

        # Calculate velocities
        u_L = hu_L / h_L if h_L > 0 else 0
        v_L = hv_L / h_L if h_L > 0 else 0
        u_R = hu_R / h_R if h_R > 0 else 0
        v_R = hv_R / h_R if h_R > 0 else 0

        # Dot product of velocity with normal vector
        un_L = u_L * normal[0] + v_L * normal[1]
        un_R = u_R * normal[0] + v_R * normal[1]

        # Compute fluxes
        flux_L = np.array([
            h_L * un_L,
            h_L * un_L ** 2 + 0.5 * self.g * h_L ** 2,
            h_L * un_L * v_L
        ])
        flux_R = np.array([
            h_R * un_R,
            h_R * un_R ** 2 + 0.5 * self.g * h_R ** 2,
            h_R * un_R * v_R
        ])

        # Upwind flux (based on velocity direction)
        if un_L >= 0:  # Flow to the right
            return flux_L
        elif un_R <= 0:  # Flow to the left
            return flux_R
        else:  # Middle state (Riemann solver)
            return 0.5 * (flux_L + flux_R)

    def compute_source(self, i):
        n = self.n
        g = self.g
        dzb_dx, dzb_dy = self.mesh['slopes'][i]
        h, hu, hv = self.U[i]

        u = hu / h if h > 0 else 0
        v = hv / h if h > 0 else 0
        tau_b_x = (n ** 2 * u * np.sqrt(u ** 2 + v ** 2)) / (h ** (4 / 3) + 1e-8)  # Add small value to prevent division by zero
        tau_b_y = (n ** 2 * v * np.sqrt(u ** 2 + v ** 2)) / (h ** (4 / 3) + 1e-8)
        source_x = -g * h * (dzb_dx + tau_b_x)
        source_y = -g * h * (dzb_dy + tau_b_y)
        return 0, source_x, source_y

    def update_solution(self, dt):
        """Update the solution using the finite volume method."""
        num_elements = len(self.mesh['cells'])
        for i in range(num_elements):
            neighbors = self.mesh['neighbors'][i]
            element = self.mesh['cells'][i]

            # Loop over each edge of the triangle
            for j, neighbor in enumerate(neighbors):
                # Get left and right states
                U_left = self.U[i]
                U_right = np.zeros_like(U_left)
                if neighbor == -1:  # Boundary condition (e.g., wall )
                    U_right[0] = self.U[i][0]
                    U_right[1] = -self.U[i][1]
                    U_right[2] = -self.U[i][2]
                elif neighbor == -2: # inlet (fixed h, may switch to hydrograph)
                    U_right[0] = 2*self.iniH - self.U[i][0]
                    U_right[1] = 2 * self.iniQx - self.U[i][1]
                    U_right[2] = 2 * self.iniQy - self.U[i][2]
                elif neighbor == -3: # outlet
                    U_right[0] = self.U[i][0]
                    U_right[1] = self.U[i][1]
                    U_right[2] = self.U[i][2]
                else:
                    U_right = self.U[neighbor]

                # Compute normal vector for the edge
                n1, n2 = element[j], element[(j + 1) % 3]
                edge_vector = self.mesh['nodes'][n2] - self.mesh['nodes'][n1]
                normal = -np.array([-edge_vector[1], edge_vector[0]])  # Perpendicular
                normal = normal / np.linalg.norm(normal)  # Normalize

                # Compute flux across the edge
                flux = self.compute_flux(U_left, U_right, normal)
                edge_length = np.linalg.norm(edge_vector)

                # Update flux for the element
                self.F[i] += flux * edge_length / self.mesh['areas'][i]

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
                np.savetxt("output/h" + str(iter) + ".csv", oldH)
                iteration.append(iter)
                res.append(residual)
                # plt.plot(iteration, res)
                # plt.show()
                # plt.pause(.1)
                # plt.cla()