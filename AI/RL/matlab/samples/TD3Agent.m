classdef TD3Agent < handle
    properties
        % Hyperparameters
        gamma = 0.99;               % Discount factor
        tau = 0.005;                % Soft update rate
        policy_update_freq = 2;     % Delayed policy update frequency
        target_policy_noise = 0.2;  % Target policy noise
        noise_clip = 0.5;           % Noise clip range
        max_action = 1;             % Maximum action value
        min_action = -1;           % Minimum action value
        
        % Networks
        actor;                      % Actor network (policy)
        critic1;                    % Critic network (Q1)
        critic2;                    % Critic network (Q2)
        target_actor;               % Target actor network
        target_critic1;             % Target critic network 1
        target_critic2;             % Target critic network 2
        
        % Optimizer parameters
        actor_learn_rate = 1e-3;
        critic_learn_rate = 1e-3;
        
        % State and action dimensions
        state_dim;
        action_dim;
        
        % Counters
        step_count = 0;             % Step counter for delayed updates
    end
    
    methods
        function obj = TD3Agent(state_dim, action_dim, max_action)
            % Constructor
            % Inputs:
            %   state_dim: Dimension of the state space
            %   action_dim: Dimension of the action space
            %   max_action: Maximum action value
            
            obj.state_dim = state_dim;
            obj.action_dim = action_dim;
            obj.max_action = max_action;
            obj.min_action = -max_action;
            
            % Initialize networks
            obj.actor = obj.build_actor_network(state_dim, action_dim);
            obj.critic1 = obj.build_critic_network(state_dim, action_dim);
            obj.critic2 = obj.build_critic_network(state_dim, action_dim);
            
            % Initialize target networks
            obj.target_actor = obj.build_actor_network(state_dim, action_dim);
            obj.target_critic1 = obj.build_critic_network(state_dim, action_dim);
            obj.target_critic2 = obj.build_critic_network(state_dim, action_dim);
            
            % Initialize target networks with same weights
            obj.target_actor = obj.hard_update(obj.actor, obj.target_actor);
            obj.target_critic1 = obj.hard_update(obj.critic1, obj.target_critic1);
            obj.target_critic2 = obj.hard_update(obj.critic2, obj.target_critic2);
        end
        
        function actor_network = build_actor_network(obj, state_dim, action_dim)
            % Build actor network (policy)
            layers = [
                featureInputLayer(state_dim, 'Name', 'state_input')
                fullyConnectedLayer(400, 'Name', 'fc1')
                reluLayer('Name', 'relu1')
                fullyConnectedLayer(300, 'Name', 'fc2')
                reluLayer('Name', 'relu2')
                fullyConnectedLayer(action_dim, 'Name', 'output')
                tanhLayer('Name', 'tanh_output')  % Bound actions between -1 and 1
                scalingLayer('Name', 'scale_output', 'Scale', obj.max_action)  % Scale to environment's action range
            ];
            actor_network = dlnetwork(layers);
        end
        
        function critic_network = build_critic_network(obj, state_dim, action_dim)
            % Build critic network (Q-function)
            state_input = featureInputLayer(state_dim, 'Name', 'state_input');
            action_input = featureInputLayer(action_dim, 'Name', 'action_input');
            
            % State pathway
            state_path = [
                state_input
                fullyConnectedLayer(400, 'Name', 'state_fc1')
                reluLayer('Name', 'state_relu1')
                fullyConnectedLayer(300, 'Name', 'state_fc2')
            ];
            
            % Action pathway
            action_path = [
                action_input
                fullyConnectedLayer(300, 'Name', 'action_fc1')
            ];
            
            % Combined pathway
            combined_path = [
                additionLayer(2, 'Name', 'add')
                reluLayer('Name', 'combined_relu')
                fullyConnectedLayer(1, 'Name', 'q_value')
            ];
            
            % Connect layers
            critic_layers = layerGraph();
            critic_layers = addLayers(critic_layers, state_path);
            critic_layers = addLayers(critic_layers, action_path);
            critic_layers = addLayers(critic_layers, combined_path);
            
            critic_layers = connectLayers(critic_layers, 'state_fc2', 'add/in1');
            critic_layers = connectLayers(critic_layers, 'action_fc1', 'add/in2');
            
            critic_network = dlnetwork(critic_layers);
        end
        
        function action = get_action(obj, state, add_noise)
            % Get action from policy
            % Inputs:
            %   state: Current state (column vector)
            %   add_noise: Boolean flag to add exploration noise
            
            state_dl = dlarray(single(state), 'CB');
            action = predict(obj.actor, state_dl);
            action = extractdata(action)';
            
            if add_noise
                % Add exploration noise (Gaussian)
                noise = obj.target_policy_noise * randn(size(action));
                action = action + noise;
                action = min(max(action, obj.min_action), obj.max_action);  % Clip action
            end
        end
        
        function learn(obj, batch)
            % Update networks using a batch of experiences
            % Input:
            %   batch: Struct containing arrays of:
            %     - states: [state_dim x batch_size]
            %     - actions: [action_dim x batch_size]
            %     - rewards: [1 x batch_size]
            %     - next_states: [state_dim x batch_size]
            %     - dones: [1 x batch_size]
            
            % Increment step counter
            obj.step_count = obj.step_count + 1;
            
            % Convert batch data to dlarray
            states = dlarray(single(batch.states), 'CB');
            actions = dlarray(single(batch.actions), 'CB');
            rewards = dlarray(single(batch.rewards), 'CB');
            next_states = dlarray(single(batch.next_states), 'CB');
            dones = dlarray(single(batch.dones), 'CB');
            
            % -------------------- Update critics -------------------- %
            % Compute target actions with noise clipping
            target_actions = predict(obj.target_actor, next_states);
            noise = obj.target_policy_noise * randn(size(target_actions));
            noise = min(max(noise, -obj.noise_clip), obj.noise_clip);
            target_actions = target_actions + noise;
            target_actions = min(max(target_actions, obj.min_action), obj.max_action);
            
            % Compute target Q-values
            target_q1 = predict(obj.target_critic1, next_states, target_actions);
            target_q2 = predict(obj.target_critic2, next_states, target_actions);
            target_q = min(target_q1, target_q2);  % Take minimum of twin critics
            target_q = rewards + obj.gamma * (1 - dones) .* target_q;
            
            % Compute current Q-values
            current_q1 = predict(obj.critic1, states, actions);
            current_q2 = predict(obj.critic2, states, actions);
            
            % Compute critic losses (MSE)
            critic1_loss = mse(current_q1, target_q);
            critic2_loss = mse(current_q2, target_q);
            
            % Update critics using adamupdate
            [obj.critic1, ~] = adamupdate(obj.critic1, critic1_loss, [], [], obj.critic_learn_rate);
            [obj.critic2, ~] = adamupdate(obj.critic2, critic2_loss, [], [], obj.critic_learn_rate);
            
            % -------------------- Delayed policy update -------------------- %
            if mod(obj.step_count, obj.policy_update_freq) == 0
                % Compute actor loss (maximize Q1)
                actions_pred = predict(obj.actor, states);
                actor_loss = -mean(predict(obj.critic1, states, actions_pred));
                
                % Update actor using adamupdate
                [obj.actor, ~] = adamupdate(obj.actor, actor_loss, [], [], obj.actor_learn_rate);
                
                % Soft update target networks
                obj.target_actor = obj.soft_update(obj.actor, obj.target_actor, obj.tau);
                obj.target_critic1 = obj.soft_update(obj.critic1, obj.target_critic1, obj.tau);
                obj.target_critic2 = obj.soft_update(obj.critic2, obj.target_critic2, obj.tau);
            end
        end
        
        function target_net = soft_update(obj, source_net, target_net, tau)
            % Soft update target networks using Polyak averaging
            source_params = source_net.Learnables;
            target_params = target_net.Learnables;
            
            for i = 1:height(target_params)
                target_params.Value{i} = tau * source_params.Value{i} + (1 - tau) * target_params.Value{i};
            end
            
            % Update target network parameters
            target_net.Learnables = target_params;
        end
        
        function target_net = hard_update(obj, source_net, target_net)
            % Hard update target networks (copy weights)
            target_net.Learnables = source_net.Learnables;
        end
    end
end