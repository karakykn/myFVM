import meshio
import numpy as np
import matplotlib.pyplot as plt
import os

class Swe1D:
    def __init__(self, caseName, startTime=0, g=9.81, n=0.012, inlet=[0,0]):
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
        self.start = startTime
        self.inlet = inlet
        mesh_path = caseName + "/mesh/"
        self.mesh = {
            'nodes': np.loadtxt(mesh_path + 'points')[:,0],
            'cells': np.loadtxt(mesh_path + 'cells', dtype=int),
            'neighbors': np.loadtxt(mesh_path + 'neighbors', dtype=int),
            'lengths': np.loadtxt(mesh_path + 'areas'),
            'slopes': np.loadtxt(mesh_path + 'slopes'),
            'edges': np.loadtxt(mesh_path + 'edges', dtype=int)
        }
        self.g = g
        self.n = n
        self.length = np.max(self.mesh['nodes']) - np.min(self.mesh['nodes'])
        self.initialize_variables()
        self.initial_conditions()

    def initialize_variables(self):
        """Initialize conserved variables and fluxes."""
        num_elements = len(self.mesh['cells'])
        self.U = np.zeros((num_elements, 2))  # Conserved variables [h, h*u, h*v] for 1D # Sources

    def initial_conditions(self):

        self.U[:,0] = np.loadtxt(self.caseName+"/run/" + str(self.start)+'/h.csv')
        self.U[:,1] = np.loadtxt(self.caseName+"/run/" + str(self.start) + '/u.csv') * self.U[:,0]

    def update_boundaries(self):
        self.iniH = self.inlet[0]
        self.iniQx = self.inlet[1] * self.iniH

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
        if normal < 0:
            a = U_right
            U_right = U_left
            U_left = a

        h_L, hu_L = U_left
        h_R, hu_R = U_right

        # Calculate velocities
        u_L = hu_L / h_L if h_L > 0 else 0
        u_R = hu_R / h_R if h_R > 0 else 0

        # Compute fluxes
        flux_L = np.array([
            h_L * u_L,
            h_L * u_L * u_L + 0.5 * self.g * h_L ** 2
        ])
        flux_R = np.array([
            h_R * u_R,
            h_R * u_R * u_R + 0.5 * self.g * h_R ** 2
        ])

        h_L = max(h_L, 1e-8)
        h_R = max(h_R, 1e-8)
        s_L = min(u_L - np.sqrt(self.g * h_L), u_R - np.sqrt(self.g * h_R))
        s_R = max(u_L + np.sqrt(self.g * h_L), u_R + np.sqrt(self.g * h_R))
        if s_L > 0:
            return flux_L
        elif s_R < 0:
            return flux_R
        else:
            return (s_R * flux_L - s_L * flux_R + s_L * s_R * (U_right - U_left)) / (s_R - s_L)

    def compute_source(self, i):
        S_0 = self.mesh['slopes'][i]
        h, hu= self.U[i]
        u = hu / h if h > 0 else 0

        S_f = self.n ** 2 * u ** 2 / h ** (4 / 3) if h > 1e-3 else 0
        source_x = (S_0 - S_f) * self.g * h
        return 0, source_x

    def update_solution(self, dt):
        self.F = np.zeros_like(self.U)  # Fluxes
        self.S = np.zeros_like(self.U)
        edges = self.mesh['edges']
        for edge in edges:
            n2, n1 = edge[0], edge[1]
            cell = edge[2]
            neighbor = edge[3]
            U_left = self.U[cell]
            U_right = np.zeros_like(U_left)

            normal = self.mesh['nodes'][n2] - self.mesh['nodes'][n1]  # Ensure positive direction
            normal /= np.linalg.norm(normal)

            if neighbor == -1:  # Boundary condition (e.g., wall )
                U_right[0] = self.U[cell][0]
                U_right[1] = -self.U[cell][1]
            elif neighbor == -2:  # inlet (fixed h, may switch to hydrograph, or rating curve)
                U_right[0] = 2 * self.iniH - self.U[cell][0]
                U_right[1] = 2 * self.iniQx - self.U[cell][1]
            elif neighbor == -3:  # outlet
                U_right[0] = self.U[cell][0]
                U_right[1] = self.U[cell][1]
            else:
                U_right = self.U[neighbor]

            flux = self.compute_flux(U_left, U_right, normal)
            self.F[cell] += flux * normal / self.mesh['lengths'][cell]
            if neighbor >= 0:
                self.F[neighbor] -= flux * normal / self.mesh['lengths'][neighbor]

        for i in range(self.mesh['cells'].shape[0]):
            self.S[i] = self.compute_source(i)
            self.U[i] += dt * (-self.F[i] + self.S[i])

    def solve(self, t_max, dt, print_step):
        """Run the simulation."""
        t = 0
        oldU = np.zeros_like(self.U)
        iter = 0
        plt.ion()
        while t < t_max:
            oldU[:, :] = self.U[:, :]
            self.update_solution(dt)
            iter += 1
            t += dt
            if iter % print_step == 0:
                residual = np.max(np.abs(self.U[:] - oldU[:]))
                print(f"Time: {t}, Residual: {residual}")
                np.savetxt(self.caseName + "/run/h" + str(iter) + ".csv", self.U[:,0])
                plt.semilogy(t, residual, "ko")
                plt.show()
                plt.pause(.1)

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
            wave_speed = (self.g * h) ** 0.5
            dt_array = np.abs((CFL * self.mesh['lengths']) / (u + wave_speed + 1e-8))
            dt = np.min(dt_array)
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
                iteration.append(iter)
                res.append(residual)
                plt.semilogy(iteration, res)
                plt.show()
                plt.pause(.1)
                plt.cla()
        time_folder = f"{self.caseName}/run/{time:.4f}"
        os.makedirs(time_folder, exist_ok=True)
        np.savetxt(f"{time_folder}/h.csv", self.U[:, 0])
        np.savetxt(f"{time_folder}/u.csv", self.U[:, 1] / (self.U[:,0] + 1e-8))

    # def plot(self):
    #     caseName = self.caseName
    #     run_path = caseName + '/run/'
    #
    #     mesh = self.mesh
    #     time_dirs = sorted([d for d in os.listdir(run_path) if d.replace('.', '', 1).isdigit()], key=float)
    #
    #
    #     # Initialize min and max values for h and u
    #     h_min, h_max = float('inf'), float('-inf')
    #     u_min, u_max = float('inf'), float('-inf')
    #
    #     # First pass: Scan all files to determine global min and max
    #     for time in time_dirs:
    #         h_file = os.path.join(run_path, time, "h.csv")
    #         u_file = os.path.join(run_path, time, "u.csv")
    #
    #         try:
    #             h_values = np.loadtxt(h_file)
    #             u_values = np.loadtxt(u_file)
    #
    #             h_min, h_max = min(h_min, h_values.min()), max(h_max, h_values.max())
    #             u_min, u_max = min(u_min, u_values.min()), max(u_max, u_values.max())
    #
    #         except OSError:
    #             continue  # Skip if any file is missing
    #
    #     # Set the y-limits based on global min/max values
    #     h_ylim = (0, h_max * 1.2)
    #     u_ylim = (u_min * 1.2, u_max * 1.2)
    #
    #     # Set up the plot
    #     plt.ion()
    #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    #
    #     # Second pass: Read and plot the data
    #     for time in time_dirs:
    #         h_file = os.path.join(run_path, time, "h.csv")
    #         u_file = os.path.join(run_path, time, "u.csv")
    #
    #         try:
    #             h_values = np.loadtxt(h_file)
    #             u_values = np.loadtxt(u_file)
    #         except OSError:
    #             continue  # Skip if any file is missing
    #
    #         ax1.clear()
    #         ax2.clear()
    #
    #         # Plot h values
    #         for i, cell in enumerate(mesh['cells']):
    #             ax1.plot([mesh['nodes'][cell[0]], mesh['nodes'][cell[1]]], [h_values[i], h_values[i]], "b")
    #
    #         ax1.set_title(f"Time: {time}s")
    #         ax1.set_ylim(h_ylim)
    #         ax1.set_xlabel("x ($m$)")
    #         ax1.set_ylabel("Water Level ($m$)")
    #
    #         # Plot u values
    #         for i, cell in enumerate(mesh['cells']):
    #             ax2.plot([mesh['nodes'][cell[0]], mesh['nodes'][cell[1]]], [u_values[i], u_values[i]], "r")
    #
    #         ax2.set_ylim(u_ylim)
    #         ax2.set_xlabel("x ($m$)")
    #         ax2.set_ylabel("Velocity ($ms^{-1}$)")
    #
    #         plt.pause(0.1)
    #
    #     plt.ioff()
    #     plt.show()



    def plot(self):
        caseName = self.caseName
        run_path = caseName + '/run/'

        mesh = self.mesh
        time_dirs = sorted([d for d in os.listdir(run_path) if d.replace('.', '', 1).isdigit()], key=float)

        # Initialize min and max values for h and u
        h_min, h_max = float('inf'), float('-inf')
        u_min, u_max = float('inf'), float('-inf')

        # First pass: Scan all files to determine global min and max
        for time in time_dirs:
            h_file = os.path.join(run_path, time, "h.csv")
            u_file = os.path.join(run_path, time, "u.csv")

            try:
                h_values = np.loadtxt(h_file)
                u_values = np.loadtxt(u_file)

                h_min, h_max = min(h_min, h_values.min()), max(h_max, h_values.max())
                u_min, u_max = min(u_min, u_values.min()), max(u_max, u_values.max())

            except OSError:
                continue  # Skip if any file is missing

        # Set the z-limits based on global min/max values for water level (h)
        h_ylim = (0, h_max * 1.2)
        u_ylim = (u_min * 1.2, u_max * 1.2)

        # Set up the 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Second pass: Read and plot the data
        for time in time_dirs:
            h_file = os.path.join(run_path, time, "h.csv")
            u_file = os.path.join(run_path, time, "u.csv")

            try:
                h_values = np.loadtxt(h_file)
                u_values = np.loadtxt(u_file)
            except OSError:
                continue  # Skip if any file is missing

            ax.clear()

            # Plot the cells in 3D
            for i, cell in enumerate(mesh['cells']):
                # Get the node positions for the current cell (triangle)
                x1, y1 = mesh['nodes'][cell[0]]
                x2, y2 = mesh['nodes'][cell[1]]
                x3, y3 = mesh['nodes'][cell[2]]

                # Get the water level (z-coordinate) for each vertex of the triangle
                z1 = h_values[i]
                z2 = h_values[i]
                z3 = h_values[i]

                # Get the velocity magnitude for coloring (use the velocity value at the centroid of the triangle)
                u_magnitude = np.linalg.norm(u_values[i])  # Assuming u_values is 2D velocity, adjust if 3D

                # Create the triangle and color it based on velocity magnitude
                ax.plot_trisurf([x1, x2, x3], [y1, y2, y3], [z1, z2, z3],
                                color=plt.cm.viridis((u_magnitude - u_min) / (u_max - u_min)))

            ax.set_title(f"Time: {time}s")
            ax.set_xlabel("x ($m$)")
            ax.set_ylabel("y ($m$)")
            ax.set_zlabel("Water Level ($m$)")
            ax.set_zlim(h_ylim)

            plt.pause(0.1)

        plt.ioff()
        plt.show()
