from source import *

caseName = "1dDambreak"
solver = Swe1D(caseName, qx_ini=0, h_ini=0.03)
# solver.solve(t_max=1.0, dt=0.01)
solver.iterativeSolve(0.01, 1e-8, 50)