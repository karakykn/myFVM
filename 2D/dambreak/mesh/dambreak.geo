lc = 5;

Point(1) = {0, 0, 0, lc};
Point(2) = {2000, 0,  0, lc};
Point(3) = {2000, 100, 0, lc};
Point(4) = {0,  100, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1,2,3,4};

Plane Surface(1) = {1};

Physical Curve("wall") = {1, 3};
Physical Curve("inlet") = {4};
Physical Curve("outlet") = {2};
Physical Surface("Domain") = {1};


