classdef TwoRManipulatorEnv < handle
    properties
        % Robot parameters
        l1 = 1;         % Length of link 1
        l2 = 1;         % Length of link 2
        q1 = 0;         % Joint angle 1 (rad)
        q2 = 0;         % Joint angle 2 (rad)
        q1_dot = 0;     % Joint velocity 1
        q2_dot = 0;     % Joint velocity 2
        
        % Target and obstacle
        targetPos = [1.5; 0.5];     % Target position [x; y]
        obstaclePos = [0.5; 0.5];   % Obstacle position [x; y]
        obstacleRadius = 0.2;       % Obstacle radius
        targetTolerance = 0.05;     % Target tolerance radius
        
        % Visualization
        fig;
        ax;
        robotPlot;
        targetPlot;
        obstaclePlot;
        eefPlot;
        
        % RL parameters
        maxJointVel = 2;    % Max joint velocity (rad/s)
        dt = 0.05;          % Time step
    end
    
    methods
        function obj = TwoRManipulatorEnv()
            % Initialize environment
            obj.reset();
        end
        
        function state = reset(obj)
            % Reset to random initial configuration
            obj.q1 = rand() * 2*pi;
            obj.q2 = rand() * 2*pi;
            obj.q1_dot = 0;
            obj.q2_dot = 0;
            
            % Ensure target is reachable
            L = obj.l1 + obj.l2;
            obj.targetPos = (rand(2,1)*2 - 1) * L * 0.8;
            
            % Ensure obstacle is between start and target
            obj.obstaclePos = (rand(2,1)*2 - 1) * L * 0.5;
            
            state = obj.getState();
        end
        
        function [nextState, reward, isDone, info] = step(obj, action)
            % Apply action (joint velocities)
            obj.q1_dot = max(min(action(1), obj.maxJointVel), -obj.maxJointVel);
            obj.q2_dot = max(min(action(2), obj.maxJointVel), -obj.maxJointVel);
            
            % Update joint angles
            obj.q1 = obj.q1 + obj.q1_dot * obj.dt;
            obj.q2 = obj.q2 + obj.q2_dot * obj.dt;
            
            % Normalize angles
            obj.q1 = mod(obj.q1, 2*pi);
            obj.q2 = mod(obj.q2, 2*pi);
            
            % Get new state
            nextState = obj.getState();
            
            % Calculate reward
            eefPos = obj.getEEFPos();
            targetDist = norm(eefPos - obj.targetPos);
            obstacleDist = norm(eefPos - obj.obstaclePos);
            
            % Check if reached target
            if targetDist < obj.targetTolerance
                reward = 2000;
                isDone = true;
            % Check if collided with obstacle
            elseif obstacleDist < obj.obstacleRadius
                reward = -200;
                isDone = true;
            else
                % Shaping reward
                reward = -1 - targetDist*0.1;
                isDone = false;
            end
            
            info = struct();
        end
        
        function state = getState(obj)
            % Get EEF position
            eefPos = obj.getEEFPos();
            
            % State: [cos(q1), sin(q1), cos(q2), sin(q2), q1_dot, q2_dot, target_x, target_y, obstacle_x, obstacle_y]
            state = [cos(obj.q1); sin(obj.q1); 
                    cos(obj.q2); sin(obj.q2);
                    obj.q1_dot; obj.q2_dot;
                    obj.targetPos;
                    obj.obstaclePos];
        end
        
        function pos = getEEFPos(obj)
            % Forward kinematics
            x1 = obj.l1 * cos(obj.q1);
            y1 = obj.l1 * sin(obj.q1);
            x2 = x1 + obj.l2 * cos(obj.q1 + obj.q2);
            y2 = y1 + obj.l2 * sin(obj.q1 + obj.q2);
            pos = [x2; y2];
        end
        
        function render(obj)
            % Create figure if it doesn't exist
            if isempty(obj.fig) || ~isvalid(obj.fig)
                obj.fig = figure(1);
                clf;
                obj.ax = gca;
                hold(obj.ax, 'on');
                axis(obj.ax, 'equal');
                xlim(obj.ax, [-2, 2]);
                ylim(obj.ax, [-2, 2]);
                grid(obj.ax, 'on');
                title(obj.ax, '2R Manipulator Environment');
            end
            
            % Clear previous plot
            if ~isempty(obj.robotPlot)
                delete(obj.robotPlot);
            end
            if ~isempty(obj.eefPlot)
                delete(obj.eefPlot);
            end
            
            % Compute link positions
            x0 = 0; y0 = 0;
            x1 = obj.l1 * cos(obj.q1);
            y1 = obj.l1 * sin(obj.q1);
            x2 = x1 + obj.l2 * cos(obj.q1 + obj.q2);
            y2 = y1 + obj.l2 * sin(obj.q1 + obj.q2);
            
            % Plot robot
            obj.robotPlot = plot(obj.ax, [x0, x1], [y0, y1], 'r', 'LineWidth', 3); % Link 1 (red)
            hold on;
            plot(obj.ax, [x1, x2], [y1, y2], 'b', 'LineWidth', 3); % Link 2 (blue)
            plot(obj.ax, x0, y0, 'yo', 'MarkerSize', 10, 'MarkerFaceColor', 'y'); % Base (yellow)
            
            % Plot EEF
            obj.eefPlot = plot(obj.ax, x2, y2, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
            
            % Plot target (if not already plotted)
            if isempty(obj.targetPlot) || ~isvalid(obj.targetPlot)
                obj.targetPlot = plot(obj.ax, obj.targetPos(1), obj.targetPos(2), 'gx', 'MarkerSize', 15, 'LineWidth', 2);
            end
            
            % Plot obstacle (if not already plotted)
            if isempty(obj.obstaclePlot) || ~isvalid(obj.obstaclePlot)
                theta = linspace(0, 2*pi, 100);
                x = obj.obstaclePos(1) + obj.obstacleRadius * cos(theta);
                y = obj.obstaclePos(2) + obj.obstacleRadius * sin(theta);
                obj.obstaclePlot = fill(obj.ax, x, y, 'k', 'FaceAlpha', 0.3);
            end
            
            drawnow;
        end
        
        function obsInfo = getObservationInfo(obj)
            % Define observation space - return dimension as scalar
            obsInfo = struct();
            obsInfo.Dimension = 10; % Since state is 10x1 vector
        end
        
        function actInfo = getActionInfo(obj)
            % Define action space (joint velocities) - return dimension as scalar
            actInfo = struct();
            actInfo.Dimension = 2; % Since action is 2x1 vector
            actInfo.UpperLimit = obj.maxJointVel;
            actInfo.LowerLimit = -obj.maxJointVel;
        end
    end
end