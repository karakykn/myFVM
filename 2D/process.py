from source import *

# caseName = input("Enter the name of the case:\n")
caseName = 'release/release_upleft'
inputF = read_input(caseName)
solver = Swe2D(caseName, startTime=inputF[0], n=inputF[1], inlet=inputF[-2])
solver.iterativeSolve(CFL=inputF[2], simTime=inputF[3], print_step=inputF[4])