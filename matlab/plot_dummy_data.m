loss_table = readtable('../data/dummy_training_loss.csv');

figure;
plot(loss_table.epoch, loss_table.average_loss, '-o', 'LineWidth', 2);
xlabel('Epoch');
ylabel('Average Loss');
title('Dummy Training Loss');
grid on;