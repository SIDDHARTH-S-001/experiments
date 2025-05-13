classdef ManipulatorEnvironment < handle
    properties
        workspace_limits = [0 10; 0 10];
        obstacles = {};
        goal_position;
        initial_position;
        goal_tolerance = 0.1;
        
        link_lengths;
        num_joints;
        current_joint_angles;
        current_joint_velocities;
        current_EEF_position;
        
        joint_velocity_limits = [-1, 1]; % Bounded to ±1 rad/s as requested
        time_step = 0.1;
    end
    
    methods
        function obj = ManipulatorEnvironment(link_lengths, initial_angles, goal_pos)
            obj.link_lengths = link_lengths;
            obj.num_joints = length(link_lengths);
            obj.current_joint_angles = initial_angles;
            obj.current_joint_velocities = zeros(size(initial_angles));
            obj.goal_position = double(goal_pos(:).');
            obj.initial_position = obj.forward_kinematics(initial_angles);
            obj.current_EEF_position = obj.initial_position;
        end
        
        function state = get_state(obj)
            % Returns the complete state space including:
            % - Joint angles
            % - Joint velocities
            % - EEF position
            % - Goal position
            % - Relative vector to goal
            eef_pos = obj.current_EEF_position;
            goal_pos = obj.goal_position;
            
            state = [
                obj.current_joint_angles;    % Joint angles
                obj.current_joint_velocities; % Joint velocities
                eef_pos(:);                  % EEF position (x,y)
                goal_pos(:);                % Goal position (x,y)
                goal_pos(:) - eef_pos(:)     % Relative vector to goal
            ];
        end
        
        function generate_obstacle(obj, obstacle_type, size)
            % Generate an obstacle of a given type and size
            % Inputs:
            %   obstacle_type: 'circle', 'square', or 'polygon'
            %   size: radius (for circle), side length (for square), or scaling factor (for polygon)
            
            if nargin < 3
                size = 0.3;  % Default size if not specified
            end
            
            min_x = min([obj.initial_position(1), obj.goal_position(1)]);
            max_x = max([obj.initial_position(1), obj.goal_position(1)]);
            min_y = min([obj.initial_position(2), obj.goal_position(2)]);
            max_y = max([obj.initial_position(2), obj.goal_position(2)]);
            
            center = [min_x + rand()*(max_x-min_x), min_y + rand()*(max_y-min_y)];
        
            switch obstacle_type
                case 'square'
                    half_size = size/2;
                    vertices = [center(1)-half_size, center(2)-half_size;
                               center(1)+half_size, center(2)-half_size;
                               center(1)+half_size, center(2)+half_size;
                               center(1)-half_size, center(2)+half_size];
                case 'circle'
                    num_vertices = 20;
                    theta = linspace(0, 2*pi, num_vertices)';
                    vertices = [center(1) + size*cos(theta), center(2) + size*sin(theta)];
                case 'polygon'
                    num_vertices = randi([3, 8]);
                    angles = sort(rand(num_vertices, 1)*2*pi);
                    radii = 0.5*size + rand(num_vertices, 1)*size;
                    vertices = [center(1) + radii.*cos(angles), center(2) + radii.*sin(angles)];
            end
            obj.obstacles{end+1} = vertices;
        end
        
        function collision = check_collision(obj, joint_angles)
            collision = false;
            [link_positions, eef_pos] = obj.get_link_positions(joint_angles);
            
            for i = 1:length(link_positions)-1
                link_start = link_positions(i,:);
                link_end = link_positions(i+1,:);
                for j = 1:length(obj.obstacles)
                    if ManipulatorEnvironment.line_polygon_intersection(link_start, link_end, obj.obstacles{j})
                        collision = true;
                        return;
                    end
                end
            end
            
            for j = 1:length(obj.obstacles)
                if ManipulatorEnvironment.point_in_polygon(eef_pos, obj.obstacles{j})
                    collision = true;
                    return;
                end
            end
        end
        
        function [new_state, reward, done] = step(obj, joint_velocities)
            % Clip joint velocities to the specified limits (±1 rad/s)
            joint_velocities = max(min(joint_velocities, obj.joint_velocity_limits(2)), ...
                                  obj.joint_velocity_limits(1));
            
            % Update joint velocities
            obj.current_joint_velocities = joint_velocities;
            
            % Calculate new joint angles
            new_angles = obj.current_joint_angles + joint_velocities * obj.time_step;
            
            % Check for collision
            if obj.check_collision(new_angles)
                new_state = obj.get_state();
                reward = -10;
                done = true;
                return;
            end
            
            % Update state
            obj.current_joint_angles = new_angles;
            obj.current_EEF_position = obj.forward_kinematics(new_angles);
            
            % Calculate reward
            prev_distance = norm(obj.forward_kinematics(obj.current_joint_angles - joint_velocities * obj.time_step) - obj.goal_position);
            distance_to_goal = norm(obj.current_EEF_position - obj.goal_position);
            reward = (prev_distance - distance_to_goal);
            
            % Check termination condition
            done = distance_to_goal < obj.goal_tolerance;
            
            % Return new state
            new_state = obj.get_state();
            
            fprintf('Step: Dist=%.2f, Reward=%.2f\n', distance_to_goal, reward);
        end
        
        function success = check_goal_reached(obj)
            distance = norm(obj.current_EEF_position - obj.goal_position);
            success = distance < obj.goal_tolerance;
        end
        
        function [link_positions, eef_pos] = get_link_positions(obj, joint_angles)
            link_positions = zeros(obj.num_joints+1, 2);
            link_positions(1,:) = [0, 0];
            
            current_angle = 0;
            current_pos = [0, 0];
            
            for i = 1:obj.num_joints
                current_angle = current_angle + joint_angles(i);
                next_pos = current_pos + obj.link_lengths(i)*[cos(current_angle), sin(current_angle)];
                link_positions(i+1,:) = next_pos;
                current_pos = next_pos;
            end
            eef_pos = current_pos;
        end
        
        function eef_pos = forward_kinematics(obj, joint_angles)
            [~, eef_pos] = obj.get_link_positions(joint_angles);
            eef_pos = double(eef_pos(:).');
        end
        
        function J = compute_jacobian(obj, joint_angles)
            [link_positions, eef_pos] = obj.get_link_positions(joint_angles);
            J = zeros(2, obj.num_joints);
            
            for i = 1:obj.num_joints
                r = eef_pos - link_positions(i,:);
                J(:, i) = [-r(2); r(1)];  % 2D Jacobian
            end
        end
        
        function [state, info] = reset(obj, initial_angles)
            % Reset the environment to initial state
            if nargin < 2
                initial_angles = zeros(obj.num_joints, 1);
            end
            
            obj.current_joint_angles = initial_angles;
            obj.current_joint_velocities = zeros(obj.num_joints, 1);
            obj.current_EEF_position = obj.forward_kinematics(initial_angles);
            
            state = obj.get_state();
            info = struct();
        end
    end
    
    methods (Static)
        function intersect = line_polygon_intersection(p1, p2, polygon)
            intersect = false;
            for i = 1:size(polygon,1)
                p3 = polygon(i,:);
                p4 = polygon(mod(i,size(polygon,1))+1,:);
                
                denom = (p4(2)-p3(2))*(p2(1)-p1(1)) - (p4(1)-p3(1))*(p2(2)-p1(2));
                if denom == 0, continue; end
                
                ua = ((p4(1)-p3(1))*(p1(2)-p3(2)) - (p4(2)-p3(2))*(p1(1)-p3(1))) / denom;
                ub = ((p2(1)-p1(1))*(p1(2)-p3(2)) - (p2(2)-p1(2))*(p1(1)-p3(1))) / denom;
                
                if ua >= 0 && ua <= 1 && ub >= 0 && ub <= 1
                    intersect = true;
                    return;
                end
            end
            
            if ManipulatorEnvironment.point_in_polygon(p1, polygon) || ...
               ManipulatorEnvironment.point_in_polygon(p2, polygon)
                intersect = true;
            end
        end
        
        function inside = point_in_polygon(point, polygon)
            x = point(1); y = point(2);
            n = size(polygon,1);
            inside = false;
            
            p1x = polygon(1,1); p1y = polygon(1,2);
            for i = 1:n+1
                p2x = polygon(mod(i-1,n)+1,1);
                p2y = polygon(mod(i-1,n)+1,2);
                
                if y > min(p1y,p2y)
                    if y <= max(p1y,p2y)
                        if x <= max(p1x,p2x)
                            if p1y ~= p2y
                                xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y) + p1x;
                            end
                            if p1x == p2x || x <= xinters
                                inside = ~inside;
                            end
                        end
                    end
                end
                p1x = p2x; p1y = p2y;
            end
        end
    end
end