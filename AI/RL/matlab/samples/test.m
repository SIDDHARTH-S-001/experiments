%% Test Script for Planar 2R Manipulator RL Evaluation
clear; close all; clc;

%% 1. Initialize the Manipulator Environment
link_lengths = [1.0, 1.0]; % Two links of 1m each
initial_angles = [0; 0]; % Start at zero joint angles
goal_position = [1.5, 1.5]; % Target end-effector position

env = ManipulatorEnvironment(link_lengths, initial_angles, goal_position);

% Add some obstacles
env.generate_obstacle('circle', 0.3);
% env.generate_obstacle('square', 0.4);
% env.generate_obstacle('polygon', 0.5);

%% 2. Initialize the Dense Reward System
reward_system = DenseRewardSystem(...
    0.1, ...    % goal_tolerance (meters)
    -10, ...    % collision_penalty
    -0.1, ...   % time_penalty (per step)
    100, ...    % goal_reward
    0.2, ...    % safety_margin
    -5 ...      % proximity_penalty
);

%% 3. Initialize the TD3 Agent
% Get state dimension by checking the actual state vector size
[~, ~] = env.reset(initial_angles);
sample_state = env.get_state();
state_dim = length(sample_state);
action_dim = env.num_joints; % Action vector dimension (joint velocities)
max_action = 1; % Maximum joint velocity (rad/s)

agent = TD3Agent(state_dim, action_dim, max_action);

%% 4. Training Parameters
max_episodes = 100;
max_steps = 200;
batch_size = 128;
replay_buffer = []; % Simple buffer for this demo

%% 5. Training Loop
training_stats = struct(...
    'EpisodeReward', zeros(max_episodes, 1), ...
    'EpisodeSteps', zeros(max_episodes, 1), ...
    'IsSuccess', zeros(max_episodes, 1), ...
    'EpisodeQ0', zeros(max_episodes, 1) ...
);

for episode = 1:max_episodes
    % Reset environment
    [state, ~] = env.reset(initial_angles);
    episode_reward = 0;
    episode_q0 = 0;
    steps = 0;
    done = false;
    
    while ~done && steps < max_steps
        % Get action from policy (with exploration noise)
        action = agent.get_action(state, true);
        
        % Step environment
        [next_state, reward, done] = env.step(action);
        
        % Store transition in replay buffer
        transition = struct(...
            'state', state, ...
            'action', action, ...
            'reward', reward, ...
            'next_state', next_state, ...
            'done', done ...
        );
        
        if isempty(replay_buffer)
            replay_buffer = transition;
        else
            replay_buffer(end+1) = transition;
        end
        
        % Sample a batch from replay buffer
        if length(replay_buffer) >= batch_size
            batch_idx = randperm(length(replay_buffer), min(batch_size, length(replay_buffer)));
            batch = struct(...
                'states', cat(2, replay_buffer(batch_idx).state), ...
                'actions', cat(2, replay_buffer(batch_idx).action), ...
                'rewards', [replay_buffer(batch_idx).reward], ...
                'next_states', cat(2, replay_buffer(batch_idx).next_state), ...
                'dones', [replay_buffer(batch_idx).done] ...
            );
            
            % Train agent with the batch
            agent.learn(batch);
            
            % Store Q-value for monitoring
            state_dl = dlarray(single(batch.states(:,1)), 'CB');
            action_dl = dlarray(single(batch.actions(:,1)), 'CB');
            q_value = predict(agent.critic1, state_dl, action_dl);
            episode_q0 = episode_q0 + extractdata(q_value);
        end
        
        % Update for next iteration
        state = next_state;
        episode_reward = episode_reward + reward;
        steps = steps + 1;
    end
    
    % Record episode statistics
    training_stats.EpisodeReward(episode) = episode_reward;
    training_stats.EpisodeSteps(episode) = steps;
    training_stats.IsSuccess(episode) = env.check_goal_reached();
    training_stats.EpisodeQ0(episode) = episode_q0 / max(1, steps);
    
    % Display progress
    fprintf('Episode %d: Reward=%.2f, Steps=%d, Success=%d, AvgQ=%.2f\n', ...
        episode, episode_reward, steps, training_stats.IsSuccess(episode), ...
        training_stats.EpisodeQ0(episode));
    
    % Periodically visualize
    if mod(episode, 10) == 0 || episode == 1
        visualize_training_progress(env, agent, episode, training_stats);
    end
end

%% 6. Evaluation
% Create a simple analyzer since the full RLManipulatorAnalyzer might need adjustments
fprintf('\n=== Evaluation ===\n');
num_eval_episodes = 10;
success_count = 0;

for eval_ep = 1:num_eval_episodes
    [state, ~] = env.reset(initial_angles);
    done = false;
    steps = 0;
    
    while ~done && steps < max_steps
        action = agent.get_action(state, false); % No exploration noise
        [state, ~, done] = env.step(action);
        steps = steps + 1;
    end
    
    if env.check_goal_reached()
        success_count = success_count + 1;
    end
    
    fprintf('Evaluation episode %d: %s, Steps: %d\n', ...
        eval_ep, string(env.check_goal_reached()), steps);
end

fprintf('Success rate: %.1f%%\n', success_count/num_eval_episodes*100);

%% Helper Function for Training Visualization
function visualize_training_progress(env, agent, episode, stats)
    % Create figure
    figure(1);
    clf;
    
    % Plot 1: Current policy performance
    subplot(2,2,1);
    [state, ~] = env.reset();
    [link_pos, eef_pos] = env.get_link_positions(env.current_joint_angles);
    
    % Draw manipulator
    plot(link_pos(:,1), link_pos(:,2), 'ko-', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    
    % Draw goal
    plot(env.goal_position(1), env.goal_position(2), 'g*', 'MarkerSize', 10);
    
    % Draw obstacles
    for i = 1:length(env.obstacles)
        obs = env.obstacles{i};
        patch(obs(:,1), obs(:,2), 'r', 'FaceAlpha', 0.3);
    end
    
    % Simulate a few steps
    for i = 1:20
        action = agent.get_action(state, false); % No noise for visualization
        [state, ~, ~] = env.step(action);
        [new_link_pos, ~] = env.get_link_positions(env.current_joint_angles);
        plot(new_link_pos(:,1), new_link_pos(:,2), 'k:', 'LineWidth', 0.5);
    end
    
    hold off;
    axis equal;
    xlim([-0.5, 2.5]);
    ylim([-0.5, 2.5]);
    title(sprintf('Policy Visualization (Ep %d)', episode));
    grid on;
    
    % Plot 2: Reward progression
    subplot(2,2,2);
    plot(stats.EpisodeReward(1:episode));
    xlabel('Episode');
    ylabel('Total Reward');
    title('Reward Progression');
    grid on;
    
    % Plot 3: Success rate
    subplot(2,2,3);
    window = max(1, floor(episode/10));
    smooth_success = movmean(stats.IsSuccess(1:episode), window);
    plot(smooth_success*100);
    xlabel('Episode');
    ylabel('Success Rate (%)');
    title(sprintf('Success Rate (Smoothed, window=%d)', window));
    grid on;
    ylim([0 100]);
    
    % Plot 4: Q-values
    subplot(2,2,4);
    plot(stats.EpisodeQ0(1:episode));
    xlabel('Episode');
    ylabel('Q-value Estimate');
    title('Q-value Estimates');
    grid on;
    
    drawnow;
end