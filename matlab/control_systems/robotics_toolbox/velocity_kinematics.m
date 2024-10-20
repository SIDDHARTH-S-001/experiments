%% Problem 1
% Consider a 2-link planar robot arm with link lengths l1 = 0.5m and l2 = 0.3m. The joint angles are θ1 and θ2.
% Task: 
% Write a MATLAB function to compute the end-effector velocity given the joint velocities.
% Calculate the end-effector velocity when θ1 = π/4, θ2 = π/3, θ̇1 = 0.5 rad/s, and θ̇2 = 1 rad/s.

% Define symbolic variables (This is to obtain transformation matrices from DH table)
% syms a alpha d theta real

% Values for variables
l1 = 0.5;
l2 = 0.3;
theta1 = pi/4;
theta2 = pi/3;
theta1_dot = 0.5;
theta2_dot = 1;

% syms l1 l2 theta1 theta2 theta1_dot theta2_dot

% DH parameters
% Example DH table [a, alpha, d, theta]
% Modify the following DH table as needed for your specific robot
DH_table = [ 0     0     0    theta1;
             l1    0     0    theta2;
             l2    0     0    0 ];

% Number of rows in the DH table
num_rows = size(DH_table, 1);

% Initialize a cell array to store the transformation matrices
% Its an array of cells and cells can contain data of any type
T_matrices = cell(num_rows, 1);

% Function to compute the transformation matrix from DH parameters
T_DH = @(a, alpha, d, theta) [ cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta);
                               sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta);
                               0,           sin(alpha),             cos(alpha),            d;
                               0,           0,                      0,                     1 ];

% Loop through each row of the DH table and compute the transformation matrix
for i = 1:num_rows
    a_i = DH_table(i, 1);
    alpha_i = DH_table(i, 2);
    d_i = DH_table(i, 3);
    theta_i = DH_table(i, 4);
    
    % Compute the transformation matrix for the current row
    T_matrices{i} = T_DH(a_i, alpha_i, d_i, theta_i);
    
    % Display the transformation matrix
    % fprintf('Transformation Matrix for row %d:\n', i);
    % disp(T_matrices{i});
end

T = T_matrices{1}*T_matrices{2}*T_matrices{3};

%% Using solution 1 (Eqns 5.45 and 5.47 from JJ Craig 3rd edition)
R10 = T_matrices{1}(1:3, 1:3);
R21 = T_matrices{2}(1:3, 1:3);
R32 = T_matrices{3}(1:3, 1:3);

P10 = T_matrices{1}(1:3, 4);
P21 = T_matrices{2}(1:3, 4);
P32 = T_matrices{3}(1:3, 4);

w11 = [0 0 theta1_dot]'; % Angular velocity of joint1 wrt frame 1
v11 = [0 0 0]'; % linear velocity of origin of joint1

% As per equation 5.45 (JJ Craig 3rd edition)
w22 = (R21*w11) + [0 0 theta2_dot]'; % Angular velocity of joint2 wrt frame2
v22 =  R21*(v11 + cross(P21, w11));  % Linear velocity of origin of joint2

w33 = w22;
v33 = R32*(v22 + cross(P32, w22));

%% Using the jacobian method

% Define symbolic variables
syms theta1 theta2 dtheta1 dtheta2 l1 l2 real

% DH parameters
% Example DH table [a, alpha, d, theta]
% Modify the following DH table as needed for your specific robot
DH_table = [ 0     0     0    theta1;
             l1    0     0    theta2;
             l2    0     0    0 ];

% Number of rows in the DH table
num_rows = size(DH_table, 1);

% Initialize a cell array to store the transformation matrices
% Its an array of cells and cells can contain data of any type
T_matrices = cell(num_rows, 1);

% Function to compute the transformation matrix from DH parameters
T_DH = @(a, alpha, d, theta) [ cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta);
                               sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta);
                               0,           sin(alpha),             cos(alpha),            d;
                               0,           0,                      0,                     1 ];

% Loop through each row of the DH table and compute the transformation matrix
for i = 1:num_rows
    a_i = DH_table(i, 1);
    alpha_i = DH_table(i, 2);
    d_i = DH_table(i, 3);
    theta_i = DH_table(i, 4);
    
    % Compute the transformation matrix for the current row
    T_matrices{i} = T_DH(a_i, alpha_i, d_i, theta_i);
    
    % Display the transformation matrix
    % fprintf('Transformation Matrix for row %d:\n', i);
    % disp(T_matrices{i});
end

T = T_matrices{1}*T_matrices{2}*T_matrices{3};

P = T(1:3, 4); % position of eef
J = jacobian(P, [theta1, theta2]);

% Define specific values for theta1 and theta2
theta1_val = pi/4;  % Example value for theta1
theta2_val = pi/3;  % Example value for theta2
l1_val = 0.5;         % Example length for l1
l2_val = 0.3;         % Example length for l2
theta1_dot = 0.5;
theta2_dot = 1;

theta_dot = [theta1_dot, theta2_dot]';

% Substitute the values into the Jacobian matrix
J_evaluated = subs(J, {theta1, theta2, l1, l2}, {theta1_val, theta2_val, l1_val, l2_val});

% Convert the symjbolic result to a numeric matrix
J_numeric = double(J_evaluated);

v_linear = J_numeric * theta_dot;
