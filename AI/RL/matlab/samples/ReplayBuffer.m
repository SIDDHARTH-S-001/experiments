classdef ReplayBuffer < handle
    properties
        maxSize;
        stateDim;
        actionDim;
        states;
        actions;
        rewards;
        nextStates;
        dones;
        ptr;
        size;
    end
    
    methods
        function obj = ReplayBuffer(maxSize, stateDim, actionDim)
            obj.maxSize = maxSize;
            
            % Ensure dimensions are scalars
            if isvector(stateDim)
                obj.stateDim = prod(stateDim); % Multiply dimensions if it's a vector
            else
                obj.stateDim = stateDim;
            end
            
            if isvector(actionDim)
                obj.actionDim = prod(actionDim);
            else
                obj.actionDim = actionDim;
            end
            
            % Initialize buffers
            obj.states = zeros(obj.stateDim, maxSize);
            obj.actions = zeros(obj.actionDim, maxSize);
            obj.rewards = zeros(1, maxSize);
            obj.nextStates = zeros(obj.stateDim, maxSize);
            obj.dones = zeros(1, maxSize);
            
            obj.ptr = 1;
            obj.size = 0;
        end
        
        function store(obj, state, action, reward, nextState, done)
            % Flatten state and nextState if they're not vectors
            state = state(:);
            nextState = nextState(:);
            action = action(:);
            
            obj.states(:, obj.ptr) = state;
            obj.actions(:, obj.ptr) = action;
            obj.rewards(obj.ptr) = reward;
            obj.nextStates(:, obj.ptr) = nextState;
            obj.dones(obj.ptr) = done;
            
            obj.ptr = mod(obj.ptr, obj.maxSize) + 1;
            obj.size = min(obj.size + 1, obj.maxSize);
        end
        
        function [states, actions, rewards, nextStates, dones] = sample(obj, batchSize)
            idx = randi(obj.size, [1, batchSize]);
            
            states = obj.states(:, idx);
            actions = obj.actions(:, idx);
            rewards = obj.rewards(idx);
            nextStates = obj.nextStates(:, idx);
            dones = obj.dones(idx);
        end
    end
end