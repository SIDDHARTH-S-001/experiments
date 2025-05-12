classdef DenseRewardSystem
    % DENSEREWARDSYSTEM Class for calculating dense rewards in RL/robotics tasks.
    %   Provides reward functions for goal-reaching, collision penalties, time penalties,
    %   and obstacle proximity penalties.
    
    properties
        goal_tolerance;      % Tolerance for considering EEF at goal (e.g., 0.01 meters)
        collision_penalty;   % Penalty for collision (e.g., -200)
        time_penalty;        % Penalty per time step (e.g., -1)
        goal_reward;         % Reward for reaching goal (e.g., +2000)
        safety_margin;       % Minimum safe distance to obstacles (e.g., 0.05 meters)
        proximity_penalty;   % Penalty scaling factor for proximity to obstacles (e.g., -10)
    end
    
    methods
        function obj = DenseRewardSystem(goal_tol, collision_pen, time_pen, goal_rew, safety_marg, prox_pen)
            % Constructor to initialize reward system parameters.
            % Inputs:
            %   goal_tol: Goal tolerance threshold.
            %   collision_pen: Penalty for collision.
            %   time_pen: Penalty per time step.
            %   goal_rew: Reward for reaching goal.
            %   safety_marg: Safety margin for obstacle proximity.
            %   prox_pen: Scaling factor for proximity penalty.
            
            obj.goal_tolerance = goal_tol;
            obj.collision_penalty = collision_pen;
            obj.time_penalty = time_pen;
            obj.goal_reward = goal_rew;
            obj.safety_margin = safety_marg;
            obj.proximity_penalty = prox_pen;
        end
        
        function reward = calculateReward(obj, eef_position, goal_position, is_collision, min_obstacle_distance)
            % Calculate the total reward based on conditions.
            % Inputs:
            %   eef_position: End-effector position [x, y, z].
            %   goal_position: Goal position [x, y, z].
            %   is_collision: Boolean indicating collision (true/false).
            %   min_obstacle_distance: Minimum distance to any obstacle.
            % Output:
            %   reward: Total calculated reward.
            
            reward = 0;
            
            % 1) Check if EEF is within goal tolerance and no collision
            if norm(eef_position - goal_position) <= obj.goal_tolerance && ~is_collision
                reward = reward + obj.goal_reward;
            end
            
            % 2) Apply collision penalty
            if is_collision
                reward = reward + obj.collision_penalty;
            end
            
            % 3) Apply time penalty (encourage efficiency)
            reward = reward + obj.time_penalty;
            
            % 4) Apply proximity penalty to encourage safety margin
            if min_obstacle_distance < obj.safety_margin
                % Penalize inversely proportional to distance (closer = higher penalty)
                proximity_factor = (obj.safety_margin - min_obstacle_distance) / obj.safety_margin;
                reward = reward + obj.proximity_penalty * proximity_factor;
            end
        end
        
        function setGoalTolerance(obj, new_tol)
            % Update goal tolerance.
            obj.goal_tolerance = new_tol;
        end
        
        function setCollisionPenalty(obj, new_penalty)
            % Update collision penalty.
            obj.collision_penalty = new_penalty;
        end
        
        % Add other setters/getters as needed...
    end
end