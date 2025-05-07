function train_2r_arm()
    % Create environment
    env = TwoRArmEnv();
    
    % Create policy network
    policy = PolicyNetwork();
    
    % Training parameters
    num_episodes = 1000;
    max_steps = 500;
    learning_rate = 0.001;
    gamma = 0.99;  % Discount factor
    
    % Tracking
    episode_rewards = zeros(num_episodes, 1);
    
    for episode = 1:num_episodes
        % Reset environment
        state = env.reset();
        done = false;
        total_reward = 0;
        
        % Storage for episode data
        states = zeros(max_steps, length(state));
        actions = zeros(max_steps, 2);
        rewards = zeros(max_steps, 1);
        
        step = 0;
        
        while ~done && step < max_steps
            step = step + 1;
            
            % Get action from policy
            action = policy.predict(state);
            
            % Add some exploration noise
            noise = 0.1 * randn(size(action));
            action = action + noise;
            
            % Clip action to reasonable limits
            action = max(min(action, 2), -2);  % Max joint velocity of ±2 rad/s
            
            % Take step in environment
            [next_state, done, reward] = env.step(action');
            
            % Store transition
            states(step, :) = state;
            actions(step, :) = action';
            rewards(step) = reward;
            
            % Update state and total reward
            state = next_state;
            total_reward = total_reward + reward;
            
            % Render (comment out for faster training)
            env.render();
            pause(0.01);
        end
        
        % Calculate discounted rewards
        discounted_rewards = zeros(step, 1);
        running_add = 0;
        
        for t = step:-1:1
            running_add = running_add * gamma + rewards(t);
            discounted_rewards(t) = running_add;
        end
        
        % Normalize discounted rewards
        discounted_rewards = discounted_rewards - mean(discounted_rewards);
        discounted_rewards = discounted_rewards / (std(discounted_rewards) + eps);
        
        % Update policy
        policy.update(states(1:step, :), actions(1:step, :), discounted_rewards, learning_rate);
        
        % Store episode reward
        episode_rewards(episode) = total_reward;
        
        % Display progress
        fprintf('Episode %d, Total Reward: %.1f, Steps: %d\n', ...
            episode, total_reward, step);
        
        % Plot rewards
        if mod(episode, 10) == 0
            figure(2);
            plot(1:episode, episode_rewards(1:episode), 'b-');
            xlabel('Episode');
            ylabel('Total Reward');
            title('Training Progress');
            drawnow;
        end
    end
    
    % Save trained policy
    save('trained_policy.mat', 'policy');
end