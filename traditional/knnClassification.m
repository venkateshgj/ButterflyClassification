clear
%clc;
% Load input data from .mat file
load('/MATLAB Drive/traditional/feature1.mat'); % Assumes the file contains a cell array called 'data' with one cell per class
%run('featureExtraction.m');
path = '/MATLAB Drive/Assessement/images.jpg';
img = imread(path);
new_size = [256, 256];
B = imresize(img, new_size);
grayImage = im2gray(B);
nFeature(1,8) = 0;
distance = 1; % Distance between the pixels
directions = [0 1; -1 1; -1 0; -1 -1]; % Directions to consider
numGrayLevels = 256; % Number of gray levels
    
% Compute the GLCM
glcm = graycomatrix(grayImage, 'Offset', directions, 'NumLevels', numGrayLevels, 'GrayLimits', []);
contrast1 = graycoprops(glcm, 'Contrast');
contrast2 = cell2mat(struct2cell(contrast1));
contrast_mean = (contrast2(1)+contrast2(2)+contrast2(3)+contrast2(4))/4;
energy1 = graycoprops(glcm, 'Energy');
energy2 = cell2mat(struct2cell(energy1));
energy_mean = (energy2(1)+energy2(2)+energy2(3)+energy2(4))/4;
homogeneity1 = graycoprops(glcm, 'Homogeneity');
homogeneity2 = cell2mat(struct2cell(homogeneity1));
homogeneity_mean = (homogeneity2(1)+homogeneity2(2)+homogeneity2(3)+homogeneity2(4))/4;
correlation1 = graycoprops(glcm, 'Correlation');
correlation2 = cell2mat(struct2cell(correlation1));
correlation_mean = (correlation2(1)+correlation2(2)+correlation2(3)+correlation2(4))/4;

%Local Binary Pattern
    
% Compute the LBP matrix
lbp_radius = 1;
lbp_points = 8;
lbp_mat = extractLBPFeatures(grayImage, 'Radius', lbp_radius, 'NumNeighbors', lbp_points);
%Considering a standard image for calculating Correlation for each
%image in the sample
img = imread('/MATLAB Drive/traditional/resference images.jfif');
grayImage2 = im2gray(img);
%Calculating LBP matrix of reference image
lbp_mat_ref = extractLBPFeatures(grayImage2, 'Radius', lbp_radius, 'NumNeighbors', lbp_points);
% Extract texture features from the LBP matrix of original samples

energy = sum(lbp_mat_ref.^2)/numel(lbp_mat);
contrast = sum(sum(bsxfun(@minus, lbp_mat, mean(lbp_mat)).^2));
correlation = corr2(lbp_mat, lbp_mat_ref);
homogeneity = sum(1./(1+lbp_mat.^2))/numel(lbp_mat);
    
%Entering the Extracted contrast, energy, homogeneity and correlation
%features into Feature matrix

nFeature(1,1) = contrast_mean;
nFeature(1,2) = energy_mean;
nFeature(1,3) = homogeneity_mean;
nFeature(1,4) = correlation_mean;
%LBP features
nFeature(1,5) = contrast;
nFeature(1,6) = energy;
nFeature(1,7) = correlation;
nFeature(1,8) = homogeneity;
load("knnLabels.mat");
% Define the number of classes
num_classes = length(Feature);

n_arr = {'BLUE MORMON','BUCKEYE','CABBAGE WHITE','COMMON ROSE','CRIMSON PATCHED LONGWING','MONARCH','REDADMIRAL','BLACK SWALLOWTAIL'};
% Define the number of neighbors
k = 3;

test_features = nFeature;

% Concatenate the features into a single row vector
test_feature_vector = test_features(:)';

% Compute distances between the test feature vector and each class's training data
dists = zeros(num_classes, size(train_data{1}, 1));
for i = 1:num_classes
    dists(i, :) = sum(bsxfun(@minus, train_data{i}, test_feature_vector).^2, 2)';
end


% Find k nearest neighbors for the test feature vector
[~, indices] = mink(dists(:), k);

% Predict the label for the test image
predicted_labels = zeros(k, 1);
for i = 1:k
    [class_index, sample_index] = ind2sub([num_classes, size(train_data{1}, 1)], indices(i));
    predicted_labels(i) = class_index;
end
label = mode(predicted_labels);
%disp(['Predicted label: ' num2str(label)]);
accuracy = sum(predicted_labels(:) == label) / numel(predicted_labels)*100;
output = char(n_arr(label));
disp(char(n_arr(label)));