l  = 1;
h  = 1;
a  = 0.5;
cmo  = 0.0001;

l0  = 0.00666666666667;
h0 = 0.1;

zor = 0.1;
h1 = 0.2;

Point(1)  = {     0,          0,  0, h0};
Point(2)  = {     l,          0,  0, h0};
Point(3)  = {     l,          h,  0, h0};
Point(4)  = {     0,          h,  0, h0};

Point(5)  = {     0, 0.50*h + cmo,  0, h0};
Point(6)  = {0.50*l,     0.50*h,  0, h1};
Point(7) = {     0, 0.50*h - cmo,  0, h0};

Point(8)  = {0.50*l,     h,  0, h0};
Point(9)  = {0.50*l,     0,  0, h0};
Point(10)  = {l,     0.50*h,  0, h1};

Point(11)  = {     0.50*l, 0.50*h + zor,  0, h1};
Point(12)  = {     0.50*l, 0.50*h - zor,  0, h1};
Point(13)  = {     l,  0.50*h + zor,  0, h1};
Point(14)  = {     l,  0.50*h - zor,  0, h1};
Point(15)  = {     0,  0.50*h + zor+ cmo,  0, h0};
Point(16)  = {     0,  0.50*h - zor- cmo,  0, h0};

Line(1) = {1, 9};
Line(2) = {9, 2};
Line(3) = {16, 12};
Line(4) = {12, 14};
Line(5) = {7, 6};
Line(6) = {6, 10};
Line(7) = {5, 6};
Line(8) = {15, 11};
Line(9) = {11, 13};
Line(10) = {4, 8};
Line(11) = {8, 3};
Line(12) = {1, 16};
Line(13) = {9, 12};
Line(14) = {2, 14};
Line(15) = {15, 4};
Line(16) = {11, 8};
Line(17) = {13, 3};
Line(18) = {16, 7};
Line(19) = {12, 6};
Line(20) = {14, 10};
Line(21) = {5, 15};
Line(22) = {6, 11};
Line(23) = {10, 13};

Curve Loop(1) = {12, 3, -13, -1};
Plane Surface(1) = {1};
Curve Loop(2) = {13, 4, -14, -2};
Plane Surface(2) = {2};
Curve Loop(3) = {18, 5, -19, -3};
Plane Surface(3) = {3};
Curve Loop(4) = {19, 6, -20, -4};
Plane Surface(4) = {4};
Curve Loop(5) = {22, 9, -23, -6};
Plane Surface(5) = {5};
Curve Loop(6) = {21, 8, -22, -7};
Plane Surface(6) = {6};
Curve Loop(7) = {16, 11, -17, -9};
Plane Surface(7) = {7};
Curve Loop(8) = {15, 10, -16, -8};
Plane Surface(8) = {8};

Transfinite Curve {1:11} = 25;
Transfinite Curve {12:17} = 20;
Transfinite Curve {18:23} = 6;
Transfinite Surface{1:8};
Recombine Surface{1:8};

Physical Curve("bottom", 15) = {1, 2};
Physical Curve("top", 16) = {10, 11};
Physical Surface("domain", 1) = {1, 2, 4, 3, 6, 5, 7, 8};
