clc;
clear all

%This SVM consists of a soft margin with a polynomial kernel: c = 0.1 p = 2
%load optimal parameters for SVM
p = 2;
load svm_variables.mat

%load input data - eval_data
path2data = input('Please input your eval_data path: ','s');
eval = cell2mat(struct2cell(load(path2data)));

%compute g(x_eval)
kernel_eval = zeros(2000,600);

for i = 1:2000
    for j = 1:600
        kernel_eval(i,j) = ((train(:,i)') * eval(:,j) + 1)^p;
    end
end

g_eval = (alpha.*train_label)' * kernel_eval + b_optim;

%compute eval_predicted
eval_predicted = zeros(600,1);
for i = 1:600
    if g_eval(i) > 0
        eval_predicted(i) = 1;
    else
        eval_predicted(i) = -1;
    end
end

