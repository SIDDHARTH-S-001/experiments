% Initialize reward system
rewardSys = DenseRewardSystem(0.01, -200, -1, 2000, 0.05, -10);

% Example inputs
eef_pos = [0.1, 0.2, 0.3];
goal_pos = [0.1, 0.2, 0.3];
is_collision = false;
min_obstacle_dist = 0.02;  % 0.02 meters to nearest obstacle

% Calculate reward
total_reward = rewardSys.calculateReward(eef_pos, goal_pos, is_collision, min_obstacle_dist);
disp(total_reward);