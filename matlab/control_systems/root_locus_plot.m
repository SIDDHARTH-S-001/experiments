%% Resources
% https://www.youtube.com/playlist?list=PLUMWjy5jgHK3-ca6GP6PL0AgcNGHqn33f (Brian Douglas Playlist on Root Locus)
% https://www.tutorialspoint.com/control_systems/control_systems_construction_root_locus.htm
%% Basic Root Locus Plot
num = [2 5 1];
den = [1 2 3];
sys = tf(num, den)

rlocus(sys) % This function plots the root locus of the system
[r, k] = rlocus(sys) % This function extract the closed-loop poles and associated feedback gain values 

%% Closed-Loop Pole Locations for a Set of Feedback Gain Values
num1 = [0.5 0 -1];
den1 = [4 0 3  0 2];
sys1 = tf(num1, den1);

% considering a range of values for feedback gain K
k = (1:0.5:5); % a values between 1 and 5 with increments of 0.5
r = rlocus(sys1, k);
size(r) % there are 4 closed loop poles and 9 specific gain values (column)
% even trajectory of r-locus for specific gain values can be visualized
rlocus(sys1, k)
%% SISO Tool based visualization
num2 = [1 2];
den2 = [1 4 5 6 3];
sys2 = tf(num2, den2);
%% Worked out example (manual and then software generated plot)
p = [0, -1, -5];
gain = 1;
sys3 = zpk([], p, gain);

% open loop poles at 0, -1 and -5 and doesnt have any open loop zeros,
% hence P > Z. and since P = 3, no of root locus branches = 3
% centroid of asymptotes is at -2
% angle of asymptotes are 60, 180 and 300 degrees, because of which 2 root locus branches will meet jw axis
% By using the Routh array method and special case(ii), the root locus branches intersects the imaginary axis at -j√5  and +j√5
% Break away point on s = -0.473

rlocus(sys3)


