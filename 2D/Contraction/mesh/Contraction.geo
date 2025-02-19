

lc = 8e-2;



Point(1) = {0, 0, 0, lc};

Point(2) = {.55, 0,  0, lc};
Point(3) = {2, .15, 0, lc};
Point(4) = {4,  .15, 0, lc};
Point(5) = {4,  .46, 0, lc};
Point(6) = {2,  .46, 0, lc};
Point(7) = {.55,  .61, 0, lc};
Point(8) = {0,  .61, 0, lc};



Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};


Curve Loop(1) = {1,2,3,4,5,6,7,8};


Plane Surface(1) = {1};


Physical Curve("wall") = {1, 2, 3, 5, 6, 7};
Physical Curve("inlet") = {8};
Physical Curve("outlet") = {4};
Physical Surface("Domain") = {1};


