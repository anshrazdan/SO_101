% Plot ground-truth and predicted joint actions from the evaluation CSV.

tbl = readtable('../data/real_model_predictions.csv');

num_joints = 6;

for joint_idx = 1:num_joints
    true_col = sprintf('true_action_%d', joint_idx - 1);
    pred_col = sprintf('pred_action_%d', joint_idx - 1);

    figure;
    plot(tbl.sample_idx, tbl.(true_col), '-o', 'LineWidth', 1.5);
    hold on;
    plot(tbl.sample_idx, tbl.(pred_col), '-x', 'LineWidth', 1.5);
    hold off;

    xlabel('Sample Index');
    ylabel('Action Value');
    title(sprintf('Joint %d: True vs Predicted Action', joint_idx - 1));
    legend('Ground Truth', 'Predicted', 'Location', 'best');
    grid on;
end
