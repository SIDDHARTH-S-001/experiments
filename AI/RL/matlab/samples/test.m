% Initialize
link_lengths = [1, 1, 0.5];
initial_angles = [0, 0, 0];
goal_pos = [2, 2];
env = ManipulatorEnvironment(link_lengths, initial_angles, goal_pos);

% Add obstacles
env.generate_obstacle('square');
env.generate_obstacle('circle');

% Run simulation
for i = 1:100
    % Compute goal-directed control
    desired_direction = (env.goal_position - env.current_EEF_position) * 0.1;
    J = env.compute_jacobian(env.current_joint_angles);
    joint_velocities = J' * desired_direction';  % Transpose for pseudo-inverse
    
    [~, ~, done] = env.step(joint_velocities);
    if done
        disp("Goal reached in " + i + " steps!");
        break;
    end
end