% Simple MATLAB script version of antena_simple.py
% Change only these values:

center = [0 0 33];

point1 = [-260 450 -5];
a1_deg = -31.37;
b1_deg = -36.02;

point2 = [150 450 10];
a2_deg = 28.5;
b2_deg = -33.73;


a1 = deg2rad(a1_deg);
b1 = deg2rad(b1_deg);
a2 = deg2rad(a2_deg);
b2 = deg2rad(b2_deg);

v1 = [sin(a1) * cos(b1), cos(a1) * cos(b1), sin(b1)];
v2 = [sin(a2) * cos(b2), cos(a2) * cos(b2), sin(b2)];

u1 = (point1 - center) / norm(point1 - center);
u2 = (point2 - center) / norm(point2 - center);

e1_l = v1;
e2_l = v2 - dot(v2, v1) * v1;
e2_l = e2_l / norm(e2_l);
e3_l = cross(e1_l, e2_l);
E_l = [e1_l(:), e2_l(:), e3_l(:)];

e1_g = u1;
e2_g = u2 - dot(u2, u1) * u1;
e2_g = e2_g / norm(e2_g);
e3_g = cross(e1_g, e2_g);
E_g = [e1_g(:), e2_g(:), e3_g(:)];

R = E_g * E_l.';

beta0 = deg2rad(30);
R0 = [
    1 0 0;
    0 cos(beta0) -sin(beta0);
    0 sin(beta0)  cos(beta0)
];

R_err = R0.' * R;

dBeta_deg = rad2deg(asin(R_err(3, 2)));
dAlpha_deg = rad2deg(atan2(R_err(1, 2), R_err(2, 2)));
dGamma_deg = rad2deg(atan2(-R_err(3, 1), R_err(3, 3)));

fprintf('dAlpha_deg = %.6f\n', dAlpha_deg);
fprintf('dBeta_deg  = %.6f\n', dBeta_deg);
fprintf('dGamma_deg = %.6f\n', dGamma_deg);
fprintf('\nR_err =\n');
disp(R_err);
