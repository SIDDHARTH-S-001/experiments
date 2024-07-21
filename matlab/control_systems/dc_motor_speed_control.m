%% Mathematical Model of DC motor
% syms K theta_dot V Ia J L R b s
% Input -> V
% Output -> theta_dot (angular velocity of shaft)
% Magnetic field is assumed to be constant, so torque will be proportional to armature current Ia.
% T = Kt * Ia;
% back emf (e) proportional to angular velocity of shaft (theta_dot) 
% e = Kv * theta_dot;
% In SI Units, Kt = Kv, so its taken as K uniformly.

% Transfer function model (theta_dot / V) in laplace domain
% TF = K /[(J*s + b)*(L*s + R) + (K*K)];

%% Example

J = 0.01; % moment of inertia of combination (shaft + damping + load)
b = 0.1;  % viscous damping
K = 0.01; % back emf constant
R = 1;    % armature resistance
L = 0.5;  % armature inductance
s = tf('s'); % creating s as a TF variable
TF = K/((J*s+b)*(L*s+R)+K^2); % open loop TF

% State Space representation
A = [-b/J   K/J
    -K/L   -R/L];
B = [0
    1/L];
C = [1   0];
D = 0;
motor_ss = ss(A,B,C,D);

% conversion from TF to SS can also be done as below
SS = ss(TF);

%% Analysis
% For a 1-rad/sec step reference, the design criteria are the following.
% Settling time less than 2 seconds
% Overshoot less than 5%
% Steady-state error less than 1%

linearSystemAnalyzer('step', TF, 0:0.1:10) % specify that the step response plot should include data points for times from 0 to 5 seconds in steps of 0.1 seconds

% system has one pole at s = -10 and another at s= -2
% Since the one pole is 5 times more negative than the other, the slower of the two poles will dominate the dynamics. 
% That is, the pole at s = -2 primarily determines the speed of response of the system and the system behaves similarly to a first-order system
% by slower pole, we mean the pole that decays slower (less -ve real part). This is called a dominant pole.

%% Now modelling the system as an approx 1st order system, 
rP_motor = 0.1 / (0.5*s + 1);

% plot both, the original system and the approximated 1st order system, 
linearSystemAnalyzer('step', TF, rP_motor, 0:0.1:10)

% observations:
% rise time = 1.1s for 1st order approx, 1.14 for original system
% settling time = 1.96 for 1st order approx, 2.07 for original system
% A first-order approximation of our motor system is relatively accurate. 
% The primary difference can be seen at t = 0 where a second order system will have a derivative of zero, but our first-order model will not.
% With a first-order system, the settling time is equal to 4*Tau, where Tau is the time constant which in this case is approx 0.5

% use lsim command or Linear simulation plot in linearSystemAnalyzer to study system response to various inputs.
lsim(SS, rP_motor)