Author: Ismet Karakan

This code solves 1D Shallow water equation with finite volume method.

1dDambreak is a tutorial case for the code.

Make sure numpy, meshio, matplotlib, pandas modules are installed.

Use Gmsh (free mesh generator) to generate 1D mesh. Export mesh as ".msh" in the "mesh" directory under
your case directory (1dDambreak in this case), rename the exported file as "gmsh". The exported file will not have
such extension but it will be formatted as ".msh". Edit boundaryType file under mesh directory, the first value is
for first point, and the second value is for the second point defined in gmsh. Use -1 for wall, -2 for inlet, -3 for
outlet type boundaries.

Run preProcess.py. It will ask you to enter the path of exported ".msh" file
(1dDambreak/1dDam in this case). This will create the necessary files related with the mesh.