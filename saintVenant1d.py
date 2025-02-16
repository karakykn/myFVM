import numpy as np
from meshConv import *
import matplotlib.pyplot as plt



class SaintVenantSolver:
    def __init__(self, mesh, g=9.81, n=0.012):
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
        self.mesh = mesh
        self.g = g
        self.n = n
        self.initialize_variables()

    def initialize_variables(self):
        """Initialize conserved variables and fluxes."""
        num_elements = len(self.mesh['cells'])
        self.U = np.zeros((num_elements, 3))  # Conserved variables [h, h*u, h*v] for 1D
        self.F = np.zeros_like(self.U)  # Fluxes
        self.S = np.zeros_like(self.U)  # Sources

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
        source_x = g * h * (dzb_dx + tau_b_x)
        source_y = g * h * (dzb_dy + tau_b_y)
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
                if neighbor == -1:  # Boundary condition (e.g., wall or open boundary)
                    U_right = self.U[i]  # Apply reflective or transmissive BC
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


# Example mesh structure
# mesh = {
#     'nodes': np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2], [2, 2]]),  # Node coordinates
#     'elements': np.array([[0, 1, 2], [1, 3, 2], [2, 3, 5], [3, 1, 4], [3, 4, 5], [6, 5, 4]]),  # Triangle node indices
#     'neighbors': [[-1, 1, -1], [3, 2, 0], [1, 4, -1], [1, -1, 4], [3, 5, 2], [-1, 4, -1]],  # Neighbor triangles (-1 for boundary)
#     'areas': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),  # Triangle areas
#     'slopes': np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
# }

# Initialize and run the solver
mesh = read_vtk('ContractionCoarse')
solver = SaintVenantSolver(mesh)
solver.solve(t_max=1.0, dt=0.01)