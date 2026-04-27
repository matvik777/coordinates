% Simple MATLAB script version of perevod.py
% Change only these values:

a_bad_deg = 10;
b_bad_deg = 5;

d_alpha_deg = 1;
d_beta_deg = 2;
d_gamma_deg = 3;


a_bad = deg2rad(a_bad_deg);
b_bad = deg2rad(b_bad_deg);

da = deg2rad(d_alpha_deg);
db = deg2rad(d_beta_deg);
dg = deg2rad(d_gamma_deg);

v_bad = [sin(a_bad) * cos(b_bad), cos(a_bad) * cos(b_bad), sin(b_bad)];

R_alpha = [
    cos(da)  sin(da) 0;
   -sin(da)  cos(da) 0;
    0        0       1
];

R_beta = [
    1 0        0;
    0 cos(db) -sin(db);
    0 sin(db)  cos(db)
];

R_gamma = [
    cos(dg) 0 sin(dg);
    0       1 0;
   -sin(dg) 0 cos(dg)
];

R_err = R_alpha * R_beta * R_gamma;
v_nom = R_err * v_bad(:);

a_nom_deg = rad2deg(atan2(v_nom(1), v_nom(2)));
b_nom_deg = rad2deg(asin(v_nom(3)));

fprintf('a_nom_deg = %.6f\n', a_nom_deg);
fprintf('b_nom_deg = %.6f\n', b_nom_deg);
fprintf('\nv_bad =\n');
disp(v_bad);
fprintf('\nv_nom =\n');
disp(v_nom.');
fprintf('\nR_err =\n');
disp(R_err);
