clear
%clc;
% Load input data from .mat file
load('/MATLAB Drive/traditional/feature1.mat'); % Assumes the file contains a cell array called 'data' with one cell per class

% Define the number of classes
num_classes = length(Feature);

% Define the number of neighbors
k = 3;

% Split the input data into training and testing sets
train_data = cell(num_classes, 1);
test_data = cell(num_classes, 1);
for i = 1:num_classes
    features = Feature{i};
    num_samples = size(features, 1);
    train_size = round(num_samples * 0.7); % Use 70% of samples for training
    test_size = 100-train_size;
    train_data{i} = features(1:train_size, :);
    test_data{i} = features(train_size+1:end, :);
end
% Classify the test data
num_test_samples = sum(cellfun(@(x) size(x, 1), test_data));
predicted_labels = zeros(num_classes, num_test_samples);
true_labels = zeros(num_classes, num_test_samples);
%temp_labels = zeros(test_size,k);
temp_labels= cell(num_classes,1);
test_sample_count = 0;
dists = zeros(num_classes, size(train_data{1}, 1));
for i= 1:num_classes
    for j = 1:test_size
        test_vector = test_data{i}(j,:);
        for l = 1:num_classes
            dists(l, :) = sum(bsxfun(@minus, train_data{l},test_vector).^2, 2)';%Euclidian
        end
        % Find k nearest neighbors for the test feature vector
        [~, indices] = mink(dists(:), k);
        temp_labels{i}(j,:) = ceil(indices/70);
        predicted_labels(i,test_sample_count+j) = mode(temp_labels{i}(j,:));
        true_labels(i,test_sample_count+j) = i;
    end     
    %true_labels(i, test_sample_count:test_sample_count+size(test_data{i}, 1)-1) = i;
    test_sample_count = test_sample_count + size(test_data{i}, 1);
end 
sum = 0;
count = 0;
for x =1:num_classes
    for y = 1:num_test_samples
        sum = sum + 1;
        if predicted_labels(x,y) == true_labels(x,y)
            count = count + 1;
        end  
    end
end    
accuracy = (count / numel(true_labels)) *100;
disp(['Accuracy: ' num2str(accuracy) '%']);
save('knnLabels.mat','true_labels','predicted_labels','train_data','test_data');