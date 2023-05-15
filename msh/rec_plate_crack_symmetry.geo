//+ R0 d = 1/32.
//d = 1/128;

L  = 0.1;
L1 = 0.05;
H1 = 0.02;
H2 = 0.04;
eps = 0.0001;
d = H1/200;
d1 = 10*d;
Point(1) = {0., 0., 0., d};
Point(2) = {L, 0., 0., d};
Point(3) = { L, H1, 0., d};
Point(4) = { L1, H1, 0., d};
Point(5) = { 0, H1-eps, 0., d};


//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
Line(5) = {5, 1};
//+
Curve Loop(1) = {1, 2, 3, 4,5};
//+
Plane Surface(1) = {1}; 
//+
Physical Curve(10) = {1}; //bottom
//+
Physical Curve(11) = {2}; //right
//+
Physical Curve(12) = {3}; //top symmetry
//+
Physical Curve(13) = {4}; //crack
Physical Curve(14) = {5}; //left//+

Physical Surface(100) = {1};

//+
Field[1] = Box;
//+
Field[1].Thickness = 0.06;
//+
Field[1].VIn = d;
//+
Field[1].VOut = d1;
//+
Field[1].XMax = 0.1;
//+
Field[1].XMin = 0.04;
//+
Field[1].YMax = 0.02;
//+
Field[1].YMin = 0;
//+
Background Field = 0;

