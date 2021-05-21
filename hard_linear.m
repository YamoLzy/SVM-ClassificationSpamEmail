clc;
clear all

%Load data
load train.mat
load test.mat

%Pre-process
train_data = train_data';
test_data = test_data';
train = zscore(train_data)';
test = zscore(test_data)';

%Define margin type - hard margin
c = 10e6;
        
%Define kernel - hard margin with linear kernel
kernel = zeros(2000,2000);
H = zeros(2000,2000);

for i = 1:2000
    for j = 1:2000
        kernel(i,j) = train(:,i)' * train(:,j);
        H(i,j) = train_label(i) * train_label(j) * kernel(i,j);
    end
end

%Check Mercer's condition
eigen = eig(kernel);

for i = 1:2000
    if abs(eigen(i)) <= 1e-4 && eigen(i) < 0
        eigen(i) = 0;
    end
end

m = 0;
for i = 1:2000
    if eigen(i) >= 0
        m = m + 1;
    end
end

if m == 2000
    disp('Mercer condition check result: This kernel is good for SVM.')
else
    disp('Mercer condition check result: This kernel is not suitable for SVM.')
end

%Compute alpha
f = -ones(2000,1);
A = [];
b = [];
Aeq = train_label';
beq = 0;
lb = zeros(2000,1);
ub = ones(2000,1) * c;
x0 = [];
options = optimset('LargeScale','off','MaxIter',1000);
alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);

%Select support vectors
threshold = 1e-4;
support_index = [];

for i = 1:2000
    if alpha(i) > threshold
        support_index(end+1) = i;
    end
end

%compute wo and bo
w_optim = (alpha .* train_label)' * train';
b_optim = 1/train_label(support_index(1)) - w_optim * train(:,support_index(1));

%compute g(x_test)
g_test = w_optim * test + b_optim;
test_label_new = zeros(1536,1);

for i = 1:1536
    if g_test(i) > 0
        test_label_new(i) = 1;
    else
        test_label_new(i) = -1;
    end
end

%Compute test accu
n_test_correct = 0;
for i = 1:1536
    if test_label_new(i) == test_label(i)
        n_test_correct = n_test_correct + 1;
    end
end

test_accu = n_test_correct/1536;
fprintf('The accuarcy of test data is : %f\n',test_accu);

%Compute g(x_train)
g_train = w_optim * train + b_optim;
train_label_new = zeros(2000,1);

for i = 1:2000
    if g_train(i) > 0
        train_label_new(i) = 1;
    else
        train_label_new(i) = -1;
    end
end

%Compute train accu
n_train_correct = 0;
for i = 1:2000
    if train_label_new(i) == train_label(i)
        n_train_correct = n_train_correct + 1;
    end
end

train_accu = n_train_correct/2000;
fprintf('The accuarcy of train data is : %f\n',train_accu);
