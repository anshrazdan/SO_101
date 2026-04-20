filename = '../data/act_model_predictions.csv'; % change this line only
tbl = readtable(filename);

num_joints = 6;

for j = 1:num_joints
    true_col = sprintf('true_action_%d', j-1);
    pred_col = sprintf('pred_action_%d', j-1);

    figure;
    plot(tbl.(true_col), '-o', 'LineWidth', 1.5);
    hold on;
    plot(tbl.(pred_col), '-x', 'LineWidth', 1.5);
    hold off;

    xlabel('Sample Index');
    ylabel('Action Value');
    title(sprintf('Joint %d: True vs Predicted Action', j-1));
    legend('True', 'Predicted');
    grid on;
end
