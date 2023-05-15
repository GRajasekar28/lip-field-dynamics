//+ R0 d = 1/32.
//d = 1/128;

L  = 0.100;
L1 = 0.050;
H1 = 0.025;
H2 = 0.100;
eps = 0.00001;
d = L/128;
Point(1) = {0., 0., 0., d};
Point(2) = {L, 0., 0., d};
Point(3) = { L, H2, 0., d};
Point(4) = { 0, H2, 0., d};
Point(5) = { 0, H1+eps, 0., d};
Point(6) = {L1, H1, 0., d};
Point(7) = {0., H1, 0., d};


//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 1};
//+
Curve Loop(1) = {1, 2, 3, 4,5,6,7};
//+
Plane Surface(1) = {1}; 
//+
Physical Curve(10) = {1}; //bottom
//+
Physical Curve(11) = {2}; //right
//+
Physical Curve(12) = {3}; //top
//+
Physical Curve(13) = {4}; //left top
Physical Curve(14) = {5}; //crack top
Physical Curve(15) = {6}; //crack bot
Physical Curve(16) = {7}; //left bot

//+
Physical Surface(100) = {1};
