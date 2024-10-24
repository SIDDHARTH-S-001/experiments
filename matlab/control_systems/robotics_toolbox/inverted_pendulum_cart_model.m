m = 0.15; % mass of the rod
l = 0.3; % length of the rod
I = (1/12)*m*l*l; % moment of inertia of the rod about its COM
M = 0.3; % mass of the cart
g = 9.81; % acceleration due to gravity

% Equations 
% (M + m)*x'' + m*l*theta'' = u
% (I + m*l*l)*theta'' + m*l*x'' = m*g*l*theta

% State Variables
% x1 = x
% x2 = theta
% x3 = x_dot
% x4 = theta_dot
% Therefore, 
% x1_dot = x3
% x2_dot = x4
% x3_dot and x4_dot are derived from the Equations above state variables

A = [0          0                 1 0;
     0          0                 0 1;
     -m*l/(M+m) 0                 0 0;
     0          m*g*l/(I + m*l*l) 0 0];

B = [0 0 1/(M+m) -m*l/(I + m*l*l)];

C = [1 0 0 0;
     0 1 0 0];

D = [0 0];

sys = ss(A, B.', C, D.');
sys_tf = ss2tf(A, B.', C, D.');





