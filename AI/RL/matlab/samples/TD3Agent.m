classdef TD3Agent < handle
    properties
        % Networks
        actor;
        critic1;
        critic2;
        targetActor;
        targetCritic1;
        targetCritic2;
        
        % Training parameters
        gamma = 0.99;       % Discount factor
        tau = 0.005;        % Target network update rate
        policyNoise = 0.2;  % Noise for target policy smoothing
        noiseClip = 0.5;    % Noise clip
        policyDelay = 2;    % Policy update delay
        
        % Experience replay
        replayBuffer;
        batchSize = 100;
        
        % Counters
        learnStep = 0;
    end
    
    methods
        function obj = TD3Agent(obsInfo, actInfo, bufferSize)
            % Initialize replay buffer
            obj.replayBuffer = ReplayBuffer(bufferSize, obsInfo.Dimension, actInfo.Dimension);
            
            % Create networks
            obj.createNetworks(obsInfo, actInfo);
        end
        
        function createNetworks(obj, obsInfo, actInfo)
            obsDim = obsInfo.Dimension;
            actDim = actInfo.Dimension;
            
            % Actor network
            actorLayers = [
                featureInputLayer(obsDim, 'Name', 'state')
                fullyConnectedLayer(256)
                reluLayer()
                fullyConnectedLayer(256)
                reluLayer()
                fullyConnectedLayer(actDim)
                tanhLayer()
                scalingLayer('Name', 'action', 'Scale', actInfo.UpperLimit)
            ];
            obj.actor = dlnetwork(actorLayers);
            
            % Critic networks - built using layerGraph
            obj.critic1 = obj.buildCriticNetwork(obsDim, actDim);
            obj.critic2 = obj.buildCriticNetwork(obsDim, actDim);
            
            % Create target networks
            obj.targetActor = obj.copyNetwork(obj.actor);
            obj.targetCritic1 = obj.copyNetwork(obj.critic1);
            obj.targetCritic2 = obj.copyNetwork(obj.critic2);
        end
        
        function criticNet = buildCriticNetwork(obj, obsDim, actDim)
            % State path
            statePath = [
                featureInputLayer(obsDim, 'Name', 'state')
                fullyConnectedLayer(256, 'Name', 'state_fc1')
                reluLayer('Name', 'state_relu1')
                fullyConnectedLayer(256, 'Name', 'state_fc2')
            ];
            
            % Action path
            actionPath = [
                featureInputLayer(actDim, 'Name', 'action')
                fullyConnectedLayer(256, 'Name', 'action_fc1')
            ];
            
            % Common path
            commonPath = [
                concatenationLayer(1, 2, 'Name', 'concat')
                reluLayer('Name', 'common_relu1')
                fullyConnectedLayer(256, 'Name', 'common_fc1')
                reluLayer('Name', 'common_relu2')
                fullyConnectedLayer(1, 'Name', 'output')
            ];
            
            % Create layer graph
            lgraph = layerGraph(statePath);
            lgraph = addLayers(lgraph, actionPath);
            lgraph = addLayers(lgraph, commonPath);
            
            % Connect layers
            lgraph = connectLayers(lgraph, 'state_fc2', 'concat/in1');
            lgraph = connectLayers(lgraph, 'action_fc1', 'concat/in2');
            
            criticNet = dlnetwork(lgraph);
        end
        
        function targetNet = copyNetwork(obj, sourceNet)
            % Create a new network with the same architecture
            if isequal(sourceNet.Layers(1).Name, 'state') % Actor network
                targetNet = dlnetwork(sourceNet.Layers);
            else % Critic network
                targetNet = obj.buildCriticNetwork(...
                    sourceNet.Layers(1).InputSize, ... % state input size
                    sourceNet.Layers(2).InputSize);    % action input size
            end
            
            % Copy learnable parameters
            params = sourceNet.Learnables;
            for i = 1:height(params)
                layerName = params.Layer{i};
                paramName = params.Parameter{i};
                targetNet.Learnables.Value{i} = params.Value{i};
            end
        end
        
        function action = getAction(obj, state, explorationNoise)
            if nargin < 3
                explorationNoise = 0.1;
            end
            
            stateDL = dlarray(single(state), 'CB');
            action = predict(obj.actor, stateDL);
            action = extractdata(action) + explorationNoise * randn(size(action));
        end
        
        function storeExperience(obj, state, action, reward, nextState, done)
            obj.replayBuffer.store(state, action, reward, nextState, done);
        end
        
        function train(obj)
            if obj.replayBuffer.size < obj.batchSize
                return;
            end
            
            % Sample batch
            [states, actions, rewards, nextStates, dones] = ...
                obj.replayBuffer.sample(obj.batchSize);
            
            % Convert to dlarray
            statesDL = dlarray(single(states), 'CB');
            actionsDL = dlarray(single(actions), 'CB');
            nextStatesDL = dlarray(single(nextStates), 'CB');
            rewardsDL = dlarray(single(rewards'), 'CB');
            donesDL = dlarray(single(dones'), 'CB');
            
            % Critic update
            [obj.critic1, obj.critic2] = obj.updateCritics(...
                statesDL, actionsDL, rewardsDL, nextStatesDL, donesDL);
            
            % Delayed policy update
            if mod(obj.learnStep, obj.policyDelay) == 0
                obj.actor = obj.updateActor(statesDL);
                
                % Update target networks
                obj.targetActor = obj.updateTargetNetwork(obj.targetActor, obj.actor, obj.tau);
                obj.targetCritic1 = obj.updateTargetNetwork(obj.targetCritic1, obj.critic1, obj.tau);
                obj.targetCritic2 = obj.updateTargetNetwork(obj.targetCritic2, obj.critic2, obj.tau);
            end
            
            obj.learnStep = obj.learnStep + 1;
        end
        
        function [critic1, critic2] = updateCritics(obj, states, actions, rewards, nextStates, dones)
            % Target policy smoothing
            noise = obj.policyNoise * randn(size(actions));
            noise = min(max(noise, -obj.noiseClip), obj.noiseClip);
            
            nextActions = predict(obj.targetActor, nextStates) + noise;
            nextActions = min(max(nextActions, -1), 1); % Clip to action space
            
            % Target Q-values
            targetQ1 = predict(obj.targetCritic1, nextStates, nextActions);
            targetQ2 = predict(obj.targetCritic2, nextStates, nextActions);
            targetQ = min(targetQ1, targetQ2);
            targetQ = rewards + obj.gamma * (1 - dones) .* targetQ;
            
            % Update critics (simplified - in practice use automatic differentiation)
            critic1 = obj.critic1;
            critic2 = obj.critic2;
        end
        
        function actor = updateActor(obj, states)
            % Update actor (simplified - in practice use automatic differentiation)
            actor = obj.actor;
        end
        
        function targetNet = updateTargetNetwork(obj, targetNet, net, tau)
            % Update target network parameters
            targetParams = targetNet.Learnables;
            netParams = net.Learnables;
            
            for i = 1:height(targetParams)
                targetParams.Value{i} = tau * netParams.Value{i} + (1-tau) * targetParams.Value{i};
            end
            
            targetNet.Learnables = targetParams;
        end
    end
end