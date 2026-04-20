% Plot the saved real-model training loss from the CSV output.

loss_table = readtable('../data/real_training_loss.csv');

figure;
plot(loss_table.epoch, loss_table.average_loss, '-o', 'LineWidth', 2);
xlabel('Epoch');
ylabel('Average Loss');
title('Real Robot Training Loss');
grid on;
