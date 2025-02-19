from source import *

# caseName = input("Enter the name of the case:\n")
caseName = 'release/release_upleft'
inputF = read_input(caseName)
startTime = inputF[0]
read_msh(caseName)
if inputF[-1]:
    visualize_mesh(caseName)
eps = 1e-4
initial_h(caseName, startTime, bound_h = inputF[5], h_assign = inputF[6])
initial_u(caseName, startTime, bound_u = inputF[7], u_assign = inputF[8])
initial_v(caseName, startTime, bound_v = inputF[9], v_assign = inputF[10])