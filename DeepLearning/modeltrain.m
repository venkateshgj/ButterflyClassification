% Load the image data
imds = imageDatastore('/MATLAB Drive/DatasetResize', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');

% Specify the size of the input images
inputSize = [227 227 3];

% Create an augmented training set to improve model accuracy
augmentedTrainingSet = augmentedImageDatastore(inputSize, trainingSet,'ColorPreprocessing','gray2rgb');

% Load the pretrained AlexNet model
if exist('modeltrain.mat', 'file')
    % Load a previously saved model
    loaded = load('modeltrain.mat');
    net = loaded.net;
else
    % Train a new model
    net = alexnet();
    numClasses = numel(categories(imds.Labels));
    layers = [
        imageInputLayer(inputSize)
        net(2:end-3)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    opts = trainingOptions('sgdm', ...
        'MiniBatchSize', 64,...
        'MaxEpochs', 50, ...
        'InitialLearnRate', 0.001, ...
        'Plots','training-progress');
    net = trainNetwork(augmentedTrainingSet, layers, opts);
    save('alexnet.mat', 'net');
end

% Save the trained model for future use
save('modeltrain.mat', 'net');
