clc; clear; close all;

% Simulation parameters
renderMode = 'animate'; % 'none' or 'animate'
numEpisodes = 1000;
maxStepsPerEpisode = 200;
memoryBufferSize = 1e4;

% Environment parameters
env = TwoRManipulatorEnv();
obsInfo = env.getObservationInfo();
actInfo = env.getActionInfo();

% Agent parameters
agent = TD3Agent(obsInfo, actInfo, memoryBufferSize);

% Training loop
episodeRewards = zeros(numEpisodes, 1);

for ep = 1:numEpisodes
    state = env.reset();
    episodeReward = 0;
    
    for step = 1:maxStepsPerEpisode
        % Select action
        action = agent.getAction(state);
        
        % Take step
        [nextState, reward, isDone, info] = env.step(action);
        
        % Store experience
        agent.storeExperience(state, action, reward, nextState, isDone);
        
        % Train agent
        if mod(step, 2) == 0
            agent.train();
        end
        
        % Update state and reward
        state = nextState;
        episodeReward = episodeReward + reward;
        
        % Render if needed
        if strcmp(renderMode, 'animate')
            env.render();
            pause(0.01);
        end
        
        % Check termination
        if isDone
            break;
        end
    end
    
    episodeRewards(ep) = episodeReward;
    fprintf('Episode %d, Reward: %.2f, Steps: %d\n', ep, episodeReward, step);
    
    % Plot performance
    if mod(ep, 50) == 0
        plotRobot(env, agent);
        figure(2);
        plot(1:ep, episodeRewards(1:ep));
        xlabel('Episode');
        ylabel('Reward');
        title('Training Progress');
        drawnow;
    end
end

% Save trained agent
save('trainedAgent.mat', 'agent');