from source import *

caseName = "1dDambreakBenchmark"

'''Pre-Process'''
# solver.generate_mesh()

'''Process'''
solver = Swe1D(caseName, qx_ini=0, h_ini=10, n=0)
# solver.iterativeSolve(CFL=.5, simTime=50, print_step=10)

'''Post-Process'''
solver.plot()