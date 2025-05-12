classdef Planar2RManipulator
    % Planar2RManipulator - A class to simulate a 2R planar manipulator in 2D workspace.
    % Supports forward kinematics, joint velocity control, and reset functionality.
    
    properties
        l1      % Length of link 1
        l2      % Length of link 2
        q       % Current joint angles [q1; q2] (radians)
        dt      % Time step for simulation (seconds)
    end
    
    methods
        function obj = Planar2RManipulator(l1, l2, dt)
            % Constructor for Planar2RManipulator
            % Inputs:
            %   l1 - Length of link 1 (meters)
            %   l2 - Length of link 2 (meters)
            %   dt - Time step for simulation (seconds)
            obj.l1 = l1;
            obj.l2 = l2;
            obj.dt = dt;
            obj.q = [0; 0];  % Initialize at zero angles
        end
        
        function [x, y] = forward_kinematics(obj, q)
            % Compute forward kinematics (end-effector position)
            % Inputs:
            %   q - Joint angles [q1; q2] (radians)
            % Outputs:
            %   x, y - End-effector position in 2D workspace
            q1 = q(1);
            q2 = q(2);
            
            x = obj.l1 * cos(q1) + obj.l2 * cos(q1 + q2);
            y = obj.l1 * sin(q1) + obj.l2 * sin(q1 + q2);
        end
        
        function obj = step(obj, q_dot)
            % Update joint angles using joint velocity control
            % Inputs:
            %   q_dot - Joint velocities [q1_dot; q2_dot] (radians/sec)
            % Outputs:
            %   obj - Updated manipulator object
            obj.q = obj.q + q_dot * obj.dt;
        end
        
        function [x, y, q] = reset(obj, q_init)
            % Reset the manipulator to initial joint angles
            % Inputs:
            %   q_init - Initial joint angles [q1; q2] (radians)
            % Outputs:
            %   x, y - End-effector position after reset
            %   q - Current joint angles after reset
            if nargin < 2
                q_init = [0; 0];  % Default to zero angles
            end
            
            obj.q = q_init;
            [x, y] = obj.forward_kinematics(obj.q);
            q = obj.q;
        end
        
        function [x, y] = get_eef_position(obj)
            % Get current end-effector position
            % Outputs:
            %   x, y - End-effector position
            [x, y] = obj.forward_kinematics(obj.q);
        end
        
        function q = get_joint_angles(obj)
            % Get current joint angles
            % Outputs:
            %   q - Current joint angles [q1; q2]
            q = obj.q;
        end
    end
end