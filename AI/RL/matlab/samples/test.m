% TD3_test.m
clc
clear

% Environment parameters
state_dim = 3;
action_dim = 1;
max_action = 2;

% Training parameters
num_episodes = 100;
max_steps = 200;
batch_size = 128;

% Initialize agent
agent = TD3Agent(state_dim, action_dim, max_action);

% Simple replay buffer
replay_buffer.capacity = 1e5;
replay_buffer.buffer = struct('states',{},'actions',{},'rewards',{},'next_states',{},'dones',{});
replay_buffer.size = @() numel(replay_buffer.buffer);
replay_buffer.add = @(s,a,r,ns,d) add_to_buffer(replay_buffer, s,a,r,ns,d);
replay_buffer.sample = @(n) sample_from_buffer(replay_buffer, n);

% Simple environment (replace with your actual environment)
env = @(action) deal(rand(state_dim,1), rand(), rand() > 0.95);

% Training loop
for episode = 1:num_episodes
    state = rand(state_dim,1); % env.reset()
    episode_reward = 0;
    
    for step = 1:max_steps
        % Get action (convert to column vector)
        action = agent.get_action(state, true);
        
        % Environment step
        [next_state, reward, done] = env(action);
        
        % Store experience (all as column vectors)
        replay_buffer.add(state, action, reward, next_state, done);
        
        % Train if enough samples
        if replay_buffer.size() >= batch_size
            batch = replay_buffer.sample(batch_size);
            agent.learn(batch);
        end
        
        state = next_state;
        episode_reward = episode_reward + reward;
        
        if done
            break;
        end
    end
    
    fprintf('Episode %d, Reward: %.2f\n', episode, episode_reward);
end

% Helper functions
function add_to_buffer(buffer, s, a, r, ns, d)
    if numel(buffer.buffer) >= buffer.capacity
        buffer.buffer(1) = [];
    end
    buffer.buffer(end+1) = struct('states',s, 'actions',a, 'rewards',r, ...
                                 'next_states',ns, 'dones',d);
end

function batch = sample_from_buffer(buffer, n)
    idx = randperm(numel(buffer.buffer), min(n, numel(buffer.buffer)));
    samples = buffer.buffer(idx);
    
    batch.states = cat(2, samples.states);
    batch.actions = cat(2, samples.actions);
    batch.rewards = [samples.rewards];
    batch.next_states = cat(2, samples.next_states);
    batch.dones = [samples.dones];
end