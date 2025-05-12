% Initialize the manipulator
l1 = 1.0;   % Link 1 length (meters)
l2 = 0.8;   % Link 2 length (meters)
dt = 0.01;  % Time step (seconds)
robot = Planar2RManipulator(l1, l2, dt);

% Reset to initial state (optional: provide initial angles)
[x0, y0, q0] = robot.reset([pi/4; pi/6]);  % Start at q1=45°, q2=30°
fprintf('Initial EEF position: (%.2f, %.2f)\n', x0, y0);

% Simulate joint velocity control for a few steps
q_dot = [0.1; -0.05];  % Joint velocities (rad/s)
for t = 0:dt:1.0
    % Step the simulation
    robot = robot.step(q_dot);
    
    % Get current state
    [x, y] = robot.get_eef_position();
    q = robot.get_joint_angles();
    
    % Display or log data
    fprintf('t=%.2f: q1=%.2f, q2=%.2f, EEF=(%.2f, %.2f)\n', t, q(1), q(2), x, y);
end