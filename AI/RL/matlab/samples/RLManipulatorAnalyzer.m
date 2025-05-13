classdef RLManipulatorAnalyzer
    % RLManipulatorAnalyzer - Class for evaluating and visualizing RL results for a manipulator
    %
    % Methods:
    %   evaluatePolicy - Evaluate trained policy without exploration noise
    %   visualizeEpisode - Visualize manipulator, path, obstacles, and goal
    %   plotTrainingMetrics - Plot training metrics over time
    
    properties
        env % RL environment
        policy % Trained policy
        trainingStats % Training statistics
        figHandle % Figure handle for visualizations
        videoWriter % Video writer for recording episodes
    end
    
    methods
        function obj = RLManipulatorAnalyzer(environment, trainedPolicy, trainingStatistics)
            % Constructor
            % Inputs:
            %   environment - RL environment object
            %   trainedPolicy - Trained policy object
            %   trainingStatistics - Training statistics collected during training
            
            obj.env = environment;
            obj.policy = trainedPolicy;
            obj.trainingStats = trainingStatistics;
            obj.figHandle = [];
            obj.videoWriter = [];
        end
        
        function [successRate, avgSteps] = evaluatePolicy(obj, numEpisodes)
            % Evaluate the trained policy without exploration noise
            % Inputs:
            %   numEpisodes - Number of evaluation episodes
            % Outputs:
            %   successRate - Percentage of successful episodes
            %   avgSteps - Average number of steps per episode
            
            totalSteps = 0;
            successCount = 0;
            
            for ep = 1:numEpisodes
                [observation, info] = reset(obj.env);
                done = false;
                steps = 0;
                
                while ~done
                    action = obj.policy.getAction(observation); % No exploration noise
                    [observation, ~, done, info] = step(obj.env, action);
                    steps = steps + 1;
                    
                    % Optional: visualize during evaluation
                    % obj.visualizeStep(observation, action, info);
                end
                
                totalSteps = totalSteps + steps;
                
                if info.IsSuccess
                    successCount = successCount + 1;
                end
                
                fprintf('Evaluation episode %d: %s, Steps: %d\n', ...
                    ep, string(info.IsSuccess), steps);
            end
            
            successRate = successCount / numEpisodes * 100;
            avgSteps = totalSteps / numEpisodes;
            
            fprintf('Evaluation completed. Success rate: %.2f%%, Average steps: %.2f\n', ...
                successRate, avgSteps);
        end
        
        function visualizeEpisode(obj, recordVideo)
            % Visualize a complete episode
            % Inputs:
            %   recordVideo - Boolean flag to record video (optional)
            
            if nargin < 2
                recordVideo = false;
            end
            
            % Initialize figure
            if isempty(obj.figHandle) || ~isvalid(obj.figHandle)
                obj.figHandle = figure('Name', 'Manipulator Visualization', ...
                    'Position', [100, 100, 800, 600]);
            else
                figure(obj.figHandle);
                clf;
            end
            
            % Initialize video recording if requested
            if recordVideo
                obj.videoWriter = VideoWriter('manipulator_episode', 'MPEG-4');
                obj.videoWriter.FrameRate = 10;
                open(obj.videoWriter);
            end
            
            % Run episode and visualize each step
            [observation, info] = reset(obj.env);
            done = false;
            
            % Get environment visualization properties
            [manipulatorState, goal, obstacles] = obj.parseInfo(info);
            
            % Initial plot setup
            hold on;
            axis equal;
            grid on;
            xlabel('X'); ylabel('Y'); zlabel('Z');
            title('Manipulator Episode Visualization');
            
            % Plot goal and obstacles
            plot3(goal(1), goal(2), goal(3), 'g*', 'MarkerSize', 10, 'LineWidth', 2);
            for i = 1:size(obstacles, 1)
                [x,y,z] = sphere;
                surf(x*obstacles(i,4) + obstacles(i,1), ...
                     y*obstacles(i,4) + obstacles(i,2), ...
                     z*obstacles(i,4) + obstacles(i,3), ...
                     'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', 'r');
            end
            
            % Initialize trajectory plot
            trajectory = zeros(1000, 3); % Preallocate
            trajLength = 0;
            hTraj = plot3(0, 0, 0, 'b-', 'LineWidth', 1.5);
            
            while ~done
                % Get current end-effector position
                eePos = manipulatorState.endEffectorPosition;
                trajLength = trajLength + 1;
                trajectory(trajLength, :) = eePos;
                
                % Update trajectory plot
                set(hTraj, 'XData', trajectory(1:trajLength, 1), ...
                           'YData', trajectory(1:trajLength, 2), ...
                           'ZData', trajectory(1:trajLength, 3));
                
                % Visualize manipulator
                obj.drawManipulator(manipulatorState);
                
                % Get action from policy
                action = obj.policy.getAction(observation);
                
                % Step environment
                [observation, ~, done, info] = step(obj.env, action);
                [manipulatorState, ~, obstacles] = obj.parseInfo(info);
                
                % Capture frame if recording
                if recordVideo
                    frame = getframe(obj.figHandle);
                    writeVideo(obj.videoWriter, frame);
                end
                
                drawnow;
                pause(0.05); % Control visualization speed
            end
            
            % Close video writer if recording
            if recordVideo
                close(obj.videoWriter);
                obj.videoWriter = [];
                fprintf('Episode video saved as manipulator_episode.mp4\n');
            end
        end
        
        function plotTrainingMetrics(obj)
            % Plot training metrics over time
            
            figure('Name', 'Training Metrics', 'Position', [100, 100, 1200, 800]);
            
            % Episode Reward
            subplot(3, 1, 1);
            plot(obj.trainingStats.EpisodeReward);
            xlabel('Episode');
            ylabel('Reward');
            title('Episode Reward');
            grid on;
            
            % Success Rate (smoothed)
            subplot(3, 1, 2);
            windowSize = max(1, floor(length(obj.trainingStats.EpisodeReward)/20));
            smoothSuccess = movmean(obj.trainingStats.IsSuccess, windowSize);
            plot(smoothSuccess*100);
            xlabel('Episode');
            ylabel('Success Rate (%)');
            title(sprintf('Success Rate (Smoothed, window=%d)', windowSize));
            grid on;
            ylim([0 100]);
            
            % Q-value estimates
            subplot(3, 1, 3);
            if isfield(obj.trainingStats, 'EpisodeQ0')
                plot(obj.trainingStats.EpisodeQ0);
                xlabel('Episode');
                ylabel('Q-value Estimate');
                title('Q-value Estimates');
                grid on;
            else
                text(0.5, 0.5, 'Q-value data not available', ...
                    'HorizontalAlignment', 'center');
            end
        end
    end
    
    methods (Access = private)
        function drawManipulator(obj, state)
            % Helper method to draw the manipulator
            % Inputs:
            %   state - Structure containing manipulator state
            
            % Clear previous manipulator drawing (except trajectory)
            h = findobj(gca, 'Tag', 'Manipulator');
            delete(h);
            
            % Draw links
            for i = 1:size(state.jointPositions, 1)-1
                plot3([state.jointPositions(i,1), state.jointPositions(i+1,1)], ...
                      [state.jointPositions(i,2), state.jointPositions(i+1,2)], ...
                      [state.jointPositions(i,3), state.jointPositions(i+1,3)], ...
                      'k-o', 'LineWidth', 3, 'MarkerSize', 6, 'Tag', 'Manipulator');
            end
            
            % Draw end effector
            plot3(state.endEffectorPosition(1), ...
                  state.endEffectorPosition(2), ...
                  state.endEffectorPosition(3), ...
                  'bo', 'MarkerSize', 8, 'LineWidth', 2, 'Tag', 'Manipulator');
        end
        
        function [manipulatorState, goal, obstacles] = parseInfo(obj, info)
            % Helper method to parse environment info
            % This needs to be customized based on your specific environment
            
            % Example implementation - adapt to your environment's info structure
            manipulatorState.jointPositions = info.JointPositions;
            manipulatorState.endEffectorPosition = info.EndEffectorPosition;
            goal = info.GoalPosition;
            obstacles = info.ObstaclePositions; % Format: [x,y,z,radius] for each row
        end
    end
end