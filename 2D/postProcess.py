from source import *

# caseName = input("Enter the name of the case:\n")
caseName = 'release/release_lowright'
inputF = read_input(caseName)
solver = Swe2D(caseName, startTime=inputF[0], n=inputF[1])
solver.plot_2d(interactive=inputF[-3])