% Simple MATLAB script version of ogle.py
% Change only these values:

center = [0 0 33];
point = [-280 500 -8];

alpha_deg = 0;
beta_deg = 0;
gamma_deg = 0;

beta0_deg = 30;


alpha = deg2rad(alpha_deg);
beta = deg2rad(beta_deg);
gamma = deg2rad(gamma_deg);
beta0 = deg2rad(beta0_deg);

R0 = [
    1 0 0;
    0 cos(beta0) -sin(beta0);
    0 sin(beta0)  cos(beta0)
];

R_err = [
    cos(alpha) * cos(gamma) + sin(alpha) * sin(beta) * sin(gamma), ...
    sin(alpha) * cos(beta), ...
    cos(alpha) * sin(gamma) - sin(alpha) * sin(beta) * cos(gamma);
   -sin(alpha) * cos(gamma) + cos(alpha) * sin(beta) * sin(gamma), ...
    cos(alpha) * cos(beta), ...
   -sin(alpha) * sin(gamma) - cos(alpha) * sin(beta) * cos(gamma);
   -cos(beta) * sin(gamma), ...
    sin(beta), ...
    cos(beta) * cos(gamma)
];

R_total = R0 * R_err;

u = (point - center) / norm(point - center);
v = R_total.' * u(:);

a_deg = rad2deg(atan2(v(1), v(2)));
b_deg = rad2deg(asin(v(3)));

fprintf('a_deg = %.6f\n', a_deg);
fprintf('b_deg = %.6f\n', b_deg);
fprintf('\nu =\n');
disp(u);
fprintf('\nv =\n');
disp(v.');
fprintf('\nR0 =\n');
disp(R0);
fprintf('\nR_err =\n');
disp(R_err);
fprintf('\nR_total =\n');
disp(R_total);
