classdef PolicyNetwork < handle
    properties
        W1;
        W2;
        input_size = 5;  % [theta1, theta2, eef_x, eef_y, dist_to_goal, dist_to_obstacle]
        hidden_size = 32;
        output_size = 2; % [theta1_dot, theta2_dot]
    end
    
    methods
        function obj = PolicyNetwork()
            % Initialize weights with random values
            obj.W1 = randn(obj.hidden_size, obj.input_size) * 0.1;
            obj.W2 = randn(obj.output_size, obj.hidden_size) * 0.1;
        end
        
        function action = predict(obj, state)
            % Forward pass through the network
            hidden = tanh(obj.W1 * state');
            action = obj.W2 * hidden;
        end
        
        function update(obj, states, actions, advantages, learning_rate)
            % Simple policy gradient update
            
            batch_size = size(states, 1);
            
            for i = 1:batch_size
                state = states(i, :)';
                action = actions(i, :)';
                advantage = advantages(i);
                
                % Forward pass
                hidden = tanh(obj.W1 * state);
                mean_action = obj.W2 * hidden;
                
                % Compute gradients (simplified)
                dW2 = advantage * (action - mean_action) * hidden';
                dhidden = obj.W2' * (advantage * (action - mean_action)) .* (1 - hidden.^2);
                dW1 = dhidden * state';
                
                % Update weights
                obj.W1 = obj.W1 + learning_rate * dW1;
                obj.W2 = obj.W2 + learning_rate * dW2;
            end
        end
    end
end