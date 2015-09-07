clear all; clc; close all;

%params
params.max_iter = 3000;
params.num_data = 45;
params.num_class = 2;
params.lambda = 1e-4;
params.lambda = 0.1;
params.learning_rate = 0.1;


X = load('./data/train_x.txt');
Y = load('./data/train_y.txt');
[m, n] = size(X);
X = X';
Y = Y';
X = [ones(1, m); X];
% figure; hold on;
% plot(train.X(2, :), train.X(3, :), 'ko');

% w parameters initialization
w = randn(n + 1, params.num_class);

% convert real value label to label matrix
% 1, 2, 3 -> 100, 010, 001.
l = bsxfun(@(y, ypos) (y == ypos), Y', 1:params.num_class);

al = 1

for i = 1:params.max_iter
    % calculate probability
    rsp = w' * X;
    % rsp = bsxfun(@minus, rsp, max(rsp, [], 1));
    rsp = exp(rsp);
    prob = bsxfun(@rdivide, rsp, sum(rsp));

    % calculate cost
    % log_prob = log(prob);
    % idx = sub2ind(size(log_prob), Y, 1:size(log_prob, 2));
    % cost = - sum(log_prob(idx)) / m + params.lambda * 0.5 * sum(sum(w .^ 2)) * al;
    % plot(i, cost, 'bo');

    % calculate gradient
    g = - X * (l - prob') / m + params.lambda * w * al;

    % update w
    w = w - params.learning_rate * g;
end

X = load('./data/test_x.txt');
Y = load('./data/test_y.txt');
X = X';
X = [ones(1, 999); X];
Y = Y';

rsp = w' * X(:, :);
[a, prediction] = max(rsp, [], 1);
sum(prediction == Y) / size(prediction)(2)

function y = boundary(w, x)
y = (- w(1) - w(2) * x) / w(3);
end

%plot([0.2 0.4], [boundary(w(:, 1), 0.2) boundary(w(:, 1), 0.4)], 'b-');
%plot([0.2 0.4], [boundary(w(:, 2), 0.2) boundary(w(:, 2), 0.4)], 'b-');
%plot([0.2 0.4], [boundary(w(:, 3), 0.2) boundary(w(:, 3), 0.4)], 'b-');

% hold off;

