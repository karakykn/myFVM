from source import *

caseName = 'Contraction'
solver = Swe2D(caseName, qx_ini=0.06721311475409837, qy_ini=0, h_ini=0.03)
# solver.solve(t_max=1.0, dt=0.01)
solver.iterativeSolve(0.01, 1e-8, 2)