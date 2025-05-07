classdef TwoRArmEnv < handle
    properties
        % Robot parameters
        L1 = 1;         % Length of link 1
        L2 = 1;          % Length of link 2
        theta1 = 0;      % Joint angle 1 (rad)
        theta2 = 0;      % Joint angle 2 (rad)
        dt = 0.05;       % Time step
        
        % Environment parameters
        obstacle_pos = [0.5, 0.5];  % [x,y] position of obstacle
        obstacle_radius = 0.3;      % Radius of obstacle
        goal_pos = [1.2, 1.2];     % [x,y] position of goal
        goal_tolerance = 0.1;      % Distance tolerance for success
        
        % Visualization handles
        fig;
        ax;
        link1_plot;
        link2_plot;
        eef_plot;
        base_plot;
        obstacle_plot;
        goal_plot;
        
        % Episode tracking
        step_count = 0;
        max_steps = 500;
    end
    
    methods
        function obj = TwoRArmEnv()
            % Initialize the environment and visualization
            obj.init_visualization();
        end
        
        function init_visualization(obj)
            % Create figure and axes
            obj.fig = figure('Color', 'white');
            obj.ax = axes('Parent', obj.fig);
            axis(obj.ax, 'equal');
            xlim(obj.ax, [-2, 2]);
            ylim(obj.ax, [-2, 2]);
            grid(obj.ax, 'on');
            title(obj.ax, '2R Manipulator RL Environment');
            xlabel(obj.ax, 'X-axis');
            ylabel(obj.ax, 'Y-axis');
            
            % Plot base (yellow)
            hold(obj.ax, 'on');
            obj.base_plot = plot(obj.ax, 0, 0, 'o', ...
                'MarkerSize', 10, 'MarkerFaceColor', 'yellow', ...
                'MarkerEdgeColor', 'black');
            
            % Plot links (red and blue)
            [x1, y1, x2, y2] = obj.forward_kinematics();
            obj.link1_plot = plot(obj.ax, [0, x1], [0, y1], ...
                'LineWidth', 5, 'Color', 'red');
            obj.link2_plot = plot(obj.ax, [x1, x2], [y1, y2], ...
                'LineWidth', 5, 'Color', 'blue');
            
            % Plot end effector (green circle)
            obj.eef_plot = plot(obj.ax, x2, y2, 'o', ...
                'MarkerSize', 8, 'MarkerFaceColor', 'green', ...
                'MarkerEdgeColor', 'black');
            
            % Plot obstacle (black circle)
            obj.obstacle_plot = rectangle(obj.ax, ...
                'Position', [obj.obstacle_pos(1)-obj.obstacle_radius, ...
                             obj.obstacle_pos(2)-obj.obstacle_radius, ...
                             2*obj.obstacle_radius, 2*obj.obstacle_radius], ...
                'Curvature', [1, 1], ...
                'FaceColor', 'black');
            
            % Plot goal (green square)
            obj.goal_plot = plot(obj.ax, obj.goal_pos(1), obj.goal_pos(2), 's', ...
                'MarkerSize', 10, 'MarkerFaceColor', 'none', ...
                'MarkerEdgeColor', 'green', 'LineWidth', 2);
            
            hold(obj.ax, 'off');
        end
        
        function [x1, y1, x2, y2] = forward_kinematics(obj)
            % Calculate forward kinematics
            x1 = obj.L1 * cos(obj.theta1);
            y1 = obj.L1 * sin(obj.theta1);
            x2 = x1 + obj.L2 * cos(obj.theta1 + obj.theta2);
            y2 = y1 + obj.L2 * sin(obj.theta1 + obj.theta2);
        end
        
        function [state, done, reward] = step(obj, action)
            % action: [theta1_dot, theta2_dot] - joint velocities
            
            % Update joint angles
            obj.theta1 = obj.theta1 + action(1) * obj.dt;
            obj.theta2 = obj.theta2 + action(2) * obj.dt;
            
            % Keep angles within [-pi, pi]
            obj.theta1 = wrapToPi(obj.theta1);
            obj.theta2 = wrapToPi(obj.theta2);
            
            % Get current end effector position
            [~, ~, x2, y2] = obj.forward_kinematics();
            eef_pos = [x2, y2];
            
            % Calculate distances
            dist_to_goal = norm(eef_pos - obj.goal_pos);
            dist_to_obstacle = norm(eef_pos - obj.obstacle_pos) - obj.obstacle_radius;
            
            % Update visualization
            obj.update_visualization();
            
            % Check termination conditions
            obj.step_count = obj.step_count + 1;
            
            % Success condition
            if dist_to_goal < obj.goal_tolerance
                done = true;
                reward = 2000;
                state = [obj.theta1, obj.theta2, eef_pos, dist_to_goal, dist_to_obstacle];
                return;
            end
            
            % Collision condition
            if dist_to_obstacle < 0
                done = true;
                reward = -200;
                state = [obj.theta1, obj.theta2, eef_pos, dist_to_goal, dist_to_obstacle];
                return;
            end
            
            % Max steps condition
            if obj.step_count >= obj.max_steps
                done = true;
                reward = -1 * obj.max_steps; % Only step penalty
                state = [obj.theta1, obj.theta2, eef_pos, dist_to_goal, dist_to_obstacle];
                return;
            end
            
            % Normal step
            done = false;
            reward = -1; % Step penalty
            state = [obj.theta1, obj.theta2, eef_pos, dist_to_goal, dist_to_obstacle];
        end
        
        function state = reset(obj)
            % Reset to random initial configuration
            obj.theta1 = -pi + 2*pi*rand();
            obj.theta2 = -pi + 2*pi*rand();
            obj.step_count = 0;
            
            % Get initial state
            [~, ~, x2, y2] = obj.forward_kinematics();
            eef_pos = [x2, y2];
            dist_to_goal = norm(eef_pos - obj.goal_pos);
            dist_to_obstacle = norm(eef_pos - obj.obstacle_pos) - obj.obstacle_radius;
            
            state = [obj.theta1, obj.theta2, eef_pos, dist_to_goal, dist_to_obstacle];
            
            % Update visualization
            obj.update_visualization();
        end
        
        function update_visualization(obj)
            % Update robot visualization
            [x1, y1, x2, y2] = obj.forward_kinematics();
            
            set(obj.link1_plot, 'XData', [0, x1], 'YData', [0, y1]);
            set(obj.link2_plot, 'XData', [x1, x2], 'YData', [y1, y2]);
            set(obj.eef_plot, 'XData', x2, 'YData', y2);
            
            drawnow;
        end
        
        function render(obj)
            % Refresh the visualization
            drawnow;
        end
    end
end