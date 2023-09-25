clear
clc;
workspace;

selpath='/MATLAB Drive/DatasetResize/';
n_images = 100;
n_arr = {'BLUEMORMON','BUCKEYE','CABBAGEWHITE','COMMONROSE','CRIMSON','MONARCH','REDADMIRAL','SWALLOWTAIL'};
n = numel(n_arr);
Feature= cell(n,1);
data(n_images,8) = 0;
for i = 1 : n
    new_path = fullfile(selpath,n_arr(1,i));
    r = 0;
    for ii = 1:n_images
        file = strcat(num2str(ii), '.jpg');
        new_file = fullfile(new_path, file);
        pathname_char = char(new_file);

        % Convert to a string scalar
        pathname_string = string(pathname_char);
        newStr = strrep(pathname_string, "{", "");
        new_file = strrep(newStr, "}", "");
        %disp(newStr);

        disp(new_file);
        img = imread(new_file);
        grayImage = im2gray(img);
        % Defining the GLCM properties
        distance = 1; % Distance between the pixels
        directions = [0 1; -1 1; -1 0; -1 -1]; % Directions to consider
        numGrayLevels = 256; % Number of gray levels
    
        % Compute the GLCM
        glcm = graycomatrix(grayImage, 'Offset', directions, 'NumLevels', numGrayLevels, 'GrayLimits', []);

        %Obtaining mean feature values for 4 directions considered

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
        %GLCM features
        Feature{i}(ii,1) = contrast_mean;
        Feature{i}(ii,2) = energy_mean;
        Feature{i}(ii,3) = homogeneity_mean;
        Feature{i}(ii,4) = correlation_mean;
        %LBP features
        Feature{i}(ii,5) = contrast;
        Feature{i}(ii,6) = energy;
        Feature{i}(ii,7) = correlation;
        Feature{i}(ii,8) = homogeneity;
    end
end
save('feature1.mat','Feature');
