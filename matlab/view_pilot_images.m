data = readtable('../data/pilot_dataset/episode_1/metadata.csv');

disp(data(1:min(5,height(data)),:))


plot(data.state_0)
title('Joint 0 over time')
xlabel('Frame')
ylabel('Angle')