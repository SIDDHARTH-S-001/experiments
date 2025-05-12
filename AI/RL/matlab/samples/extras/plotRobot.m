function plotRobot(env, agent)
    figure(1);
    clf;
    
    % Plot environment
    env.render();
    
    % Plot additional info if needed
    title(sprintf('2R Manipulator - Episode %d', agent.learnStep));
end