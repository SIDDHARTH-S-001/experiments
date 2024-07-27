%% This script is to explore transient response analysis - Ogata Chapter 5
% Q1)
A = [-1 -1;
     6.5 0];

B = [1 1;
     1 0];

C = [1 0;
     0 1];

D = [0 0;
     0 0];

step(A, B, C, D)

%% Q2
% In this program we plot step-response curves of a system
% having two inputs (u1 and u2) and two outputs (y1 and y2)
% We shall first plot step-response curves when the input is u1
% Then we shall plot step-response curves when the input is u2 
% Enter matrices A, B, C, and D
A = [-1 -1;6.5 0];
B = [1 1;1 0];
C = [1 0;0 1];
D = [0 0;0 0];
% ***** To plot step-response curves when the input is u1, enter
% the command 'step(A,B,C,D,1)' *****
step(A,B,C,D,1)
grid
title ('Step-Response Plots: Input = u1 (u2 = 0)')
text(3.4, -0.06,'Y1')
text(3.4, 1.4,'Y2')
hold on
% ***** Next, we shall plot step-response curves when the input
% is u2. Enter the command 'step(A,B,C,D,2)' *****
step(A,B,C,D,2)
grid
title ('Step-Response Plots: Input = u2 (u1 = 0)')
text(3,0.14,'Y1')
text(2.8,1.1,'Y2')

%% Standard form of 2nd order system
wn = 5; % natural frequency
zeta = .4; % damping ratio
[num, den] = ord2(wn, zeta); % generates a standard continuous 2nd order LTI model
num = 5^2 * num;
printsys(num, den, 's');

t = 0:0.01:3;

% Obtaining step response
% step(num, den);
[y,x,t] = step(num, den, t);
plot(t, y);
grid
title (' Unit-Step Response of G(s) = 25/(s^2+4s+25)')
xlabel('t - sec')
ylabel('Output')




