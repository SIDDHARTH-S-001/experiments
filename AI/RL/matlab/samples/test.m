%% Test Script for ManipulatorEnvironment with Smaller Obstacles
clear all
close all 
clc

%% Initialize the environment
link_lengths = [2, 1.5]; % Link lengths [l1, l2]
initial_angles = [pi/4; -pi/6]; % Initial joint angles [q1; q2]
goal_position = [2.5, 2.0]; % Goal position [x, y]

env = ManipulatorEnvironment(link_lengths, initial_angles, goal_position);

%% Generate smaller obstacles (modified sizes)
env.generate_obstacle('circle', 0.2);   % Radius = 0.2m (small circle)
% env.generate_obstacle('square', 0.25);  % Side length = 0.25m (small square)
% env.generate_obstacle('polygon', 0.15); % Small polygon

%% Test 1: Verify initial state
disp('=== Test 1: Initial State ===');
initial_state = env.get_state();
fprintf('State vector size: %dx1\n', length(initial_state));
fprintf('Joint angles: [%.2f, %.2f] rad\n', initial_state(1:2));
fprintf('Joint velocities: [%.2f, %.2f] rad/s\n', initial_state(3:4));
fprintf('EEF position: [%.2f, %.2f] m\n', initial_state(5:6));
fprintf('Goal position: [%.2f, %.2f] m\n', initial_state(7:8));
fprintf('Distance to goal: %.2f m\n', norm(initial_state(9:10)));

%% Test 2: Take valid actions
disp('\n=== Test 2: Valid Actions ===');
actions = [
    0.5  -0.3;   % Slow move
    0.8  0.1;    % Faster q1
    -0.2 0.6     % Faster q2
];

for i = 1:size(actions,1)
    [new_state, reward, done] = env.step(actions(i,:)');
    fprintf('\nAction %d: [%.2f, %.2f] rad/s', i, actions(i,1), actions(i,2));
    fprintf('\nNew EEF pos: [%.2f, %.2f] m', new_state(5), new_state(6));
    fprintf('\nReward: %.2f, Done: %d\n', reward, done);
end

%% Test 3: Check collision with small obstacles
disp('\n=== Test 3: Collision Detection ===');
test_angles = [
    pi/4  -pi/6;  % Initial (no collision)
    3*pi/4  pi/2  % Likely collision
];

for i = 1:size(test_angles,1)
    collision = env.check_collision(test_angles(i,:)');
    fprintf('Angles [%.2f, %.2f] rad: Collision = %d\n', ...
            test_angles(i,1), test_angles(i,2), collision);
end

%% Test 4: Visualization with small obstacles
disp('\n=== Test 4: Visualization ===');
figure(1); clf;
hold on; axis equal; grid on;
xlim([-0.5, sum(link_lengths)+0.5]);
ylim([-0.5, sum(link_lengths)+0.5]);
title('2R Manipulator with Small Obstacles');

% Plot workspace
rectangle('Position', [env.workspace_limits(1,1), env.workspace_limits(2,1), ...
                      diff(env.workspace_limits(1,:)), diff(env.workspace_limits(2,:))], ...
          'EdgeColor', 'k', 'LineStyle', '--');

% Plot goal
plot(goal_position(1), goal_position(2), 'g*', 'MarkerSize', 10, 'LineWidth', 2);

% Plot obstacles (now smaller)
for i = 1:length(env.obstacles)
    obs = env.obstacles{i};
    fill(obs(:,1), obs(:,2), 'r', 'FaceAlpha', 0.3);
end

% Get current manipulator state
[link_pos, eef_pos] = env.get_link_positions(env.current_joint_angles);

% Plot manipulator
plot(link_pos(:,1), link_pos(:,2), 'b-o', 'LineWidth', 2);
plot(link_pos(1,1), link_pos(1,2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k'); % Base
plot(link_pos(2,1), link_pos(2,2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k'); % Joint
plot(eef_pos(1), eef_pos(2), 'mo', 'MarkerSize', 8, 'MarkerFaceColor', 'm'); % EEF

xlabel('X (m)'); ylabel('Y (m)');
legend('Workspace', 'Goal', 'Obstacles', 'Manipulator', 'Location', 'best');
hold off;

%% Test 5: Reset to initial state
disp('\n=== Test 5: Environment Reset ===');
[reset_state, ~] = env.reset(initial_angles);
fprintf('Reset to angles: [%.2f, %.2f] rad\n', reset_state(1), reset_state(2));