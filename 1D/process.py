from source import *

# caseName = "release_in_tank"
# solver = Swe1D(caseName, startTime=0, n=0)
# solver.iterativeSolve(CFL=.5, simTime=50, print_step=20)

'''Process'''
caseName = "1dDambreakBenchmark"
solver = Swe1D(caseName, startTime=0, n=0, inlet=[10,0])
# solver.iterativeSolve(CFL=.5, simTime=50, print_step=20)

'''Post-Process'''
solver.plot()