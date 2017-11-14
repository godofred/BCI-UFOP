%% BCI-UFOP
% <Description>
% <Functionality>

%  Copyright (C) 2016  Vinicius Queiroz
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%    
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
tic;

%% Preamble
clc;
fprintf('Clearing all variables and closing all open MATLAB windows...\n\n');
clear all;
close all;

%% Setup Variables (CHANGE THIS TO YOUR NEEDS)
% TODO: Change to GUI settings

filename_dataset = 'featuresdata_eegmmidb2.mat';

classifier = 'ANN'; % String telling which classifier should we use
                    %   'ANN' for Artificial Neural Networks
                    %   'SVM' for Support Vector Machines (not implemented)

time_after_cue = 1.5; % How much after the cue the window should start sliding (in seconds)
window_length = 0.5; % The length of the window (in seconds)
window_padding = 0.1; % How much the window slides through each iteration (in seconds)

% Load parameters
load_fv = 0; % Load feature vector from file?
             % Yes: <load_fv> = 1
             % No: <load_fv> = 0 (This will perform the feature extraction algorithm)

load_w = 1; % Load weights from file?
            % Yes: <load_w> = 1 (No training. Only evaluation)
            % No: <load_w> = 0 (this randomly initializes Theta matrices and then
            %   performs Backpropagation to find its optimal values)
            
load_mat_EDF = 1; % Load EDF from .mat file?
                  % Yes: <load_w> = 1 
                  % No: <load_w> = 0 
                  
% Complex Networks parameters
m = 2;
t0 = 0.005;
tq = 0.9;

% ANN parameters
hidden_layer_size = 500;   % Number of hidden units
max_iter = 50;       % Number of maximum epochs
lambda = 0;                % Regularization parameter

parameter_Matrix = [time_after_cue, window_length, window_padding, m, t0, tq, hidden_layer_size, max_iter, lambda];

%% Declaration of variables
featureVector = [];

if load_fv == 0;
    
    %% Reading EDF Database
    if(load_mat_EDF)
        load('eegmmidb_3_channels.mat');
    else
        Read_eegmmidb;
    end
    %% Complex networks variables definition
    
    x = 0:m-1;
    t=t0+x*(tq-t0)/(m-1);
    

    %% Sliding window through the whole database
    total_featExt_time = tic;
    warning('off','MATLAB:colon:nonIntegerIndex') % Suppress unrelated warning
    w = 1; % Index of the window;
    for task = 1:length(header) % Iterates through all tasks from all records
        time_single_window = tic;
        fprintf('Task %d/%d\n',task,length(header));
        numAnnotation = length(header{task}.annotation.event); %% Number of events in the current task
        
        for event = 1:numAnnotation % Iterates through each event
            start_point_sec = header{task}.annotation.starttime(event); % 
            start_point_sec = start_point_sec + time_after_cue; % Process only signals obtained starting from <time_after_cue>;

            end_point_sec = start_point_sec + window_length; % Makes a <window_length> size window;

            while end_point_sec <= (header{task}.annotation.starttime(event) + header{task}.annotation.duration(event)); % Iterates until the end of the event
                
                start_point_sam = start_point_sec.*header{task}.samplerate(1)+1; 
                end_point_sam = end_point_sec.*header{task}.samplerate(1);

                window.FCz = data{task}{1}(start_point_sam:end_point_sam); %#ok<*BDSCI>
                window.C3 = data{task}{2}(start_point_sam:end_point_sam);
                window.C4 = data{task}{3}(start_point_sam:end_point_sam);
                window.t = 1:length(window.FCz);
                window.t = window.t';

    %% Pre-processing (TODO) 


    %% Feature Extraction (Complex Networks)

                FeatExtr_eegmmidb;


                start_point_sec = start_point_sec + window_padding; % Advances window by 0.1s
                end_point_sec = start_point_sec + window_length;
                w=w+1;
            end % End while (event)
        end % End for(events)
        toc(time_single_window)
    end % End for(tasks)
    warning('on','MATLAB:colon:nonIntegerIndex') % Turn on the previous disabled warning
    fprintf('Total Feature Extraction Time: %f',toc(total_featExt_time))


%% Mean Normalization

    fprintf('\nStarting mean Normalization of the feature Vector...\n');
    tic;
    featureVector_norm = featureVector;

    mu=mean(featureVector);
    sigma=std(featureVector);
    for i=1:size(featureVector,1)
        featureVector_norm(i,:)=(featureVector(i,:)-mu)./sigma;
    end

    if strcmp(classifier, 'SVM');
        featureVector = [featureVector_norm y]; %% Feature Vector for LIBSVM traning
        clearvars -except featureVector classifier parameter_Matrix
    elseif strcmp(classifier, 'ANN');
        featureVector = featureVector_norm;
        clearvars -except featureVector y classifier parameter_Matrix
    end
    toc;


    %% Feature selection (TODO) <Recommended if results present High Variance>


    %% Dataset shuffle and separation into training, validation and test datasets (60-20-20);

    fprintf('\nStarting dataset shuffle and split...\n');

    tic;
    m = size(featureVector,1);

    rand_seed = randperm(m);
    featureVector = featureVector(rand_seed,:);

    a = ceil(m*0.6);
    b = ceil(m*0.8);
    X_train = featureVector(1:a,:);
    X_valid = featureVector(a+1:b,:);
    X_test = featureVector(b+1:end,:);


    if strcmp(classifier, 'ANN')
        y = y(rand_seed,:);
        y_train = y(1:a,:);
        y_valid = y(a+1:b,:);
        y_test = y(b+1:end,:);
        clear y;
    end

    clear featureVector;
    toc;
    fprintf('\nSaving features to file......\n');
    save(filename_dataset,'X_train','X_valid','X_test','y_train','y_valid','y_test');
else % Load feature vector from file
    fprintf('\nLoading saved features from file......\n');
    tic;
    load(filename_dataset);
    toc;
end % End if(load_fv)

%% Classifier training
if strcmp(classifier, 'ANN')
    tic;
    TrainANN_eegmmidb;
    toc;
elseif strcmp(classifier, 'SVM') %% (not imlemented)
    tic;
    TrainSVM_eegmmidb;
    toc;
end

% Report accuracy on training set

tic;
fprintf('\nPredicting on training set\n');
[pred_train,c_train,cp_train] = predict(Theta1, Theta2, X_train);

%Rest efficiency
TP_rest_train = length(find(pred_train(find(y_train==1))==1));
FP_rest_train = length(find(pred_train(find(y_train~=1))==1));
TN_rest_train = length(find(pred_train(find(y_train~=1))~=1));
FN_rest_train = length(find(pred_train(find(y_train==1))~=1));

Se_rest_train = TP_rest_train/(TP_rest_train+FN_rest_train)*100;
PPV_rest_train = TP_rest_train/(TP_rest_train+FP_rest_train)*100;
FPR_rest_train = FP_rest_train/(FP_rest_train+TN_rest_train)*100;


%Left hand motor imagery efficiency
TP_left_train = length(find(pred_train(find(y_train==2))==2));
FP_left_train = length(find(pred_train(find(y_train~=2))==2));
TN_left_train = length(find(pred_train(find(y_train~=2))~=2));
FN_left_train = length(find(pred_train(find(y_train==2))~=2));

Se_left_train = TP_left_train/(TP_left_train+FN_left_train)*100;
PPV_left_train = TP_left_train/(TP_left_train+FP_left_train)*100;
FPR_left_train = FP_left_train/(FP_left_train+TN_left_train)*100;


%Right hand motor imagery efficiency
TP_right_train = length(find(pred_train(find(y_train==3))==3));
FP_right_train = length(find(pred_train(find(y_train~=3))==3));
TN_right_train = length(find(pred_train(find(y_train~=3))~=3));
FN_right_train = length(find(pred_train(find(y_train==3))~=3));

Se_right_train = TP_right_train/(TP_right_train+FN_right_train)*100;
PPV_right_train = TP_right_train/(TP_right_train+FP_right_train)*100;
FPR_right_train = FP_right_train/(FP_right_train+TN_right_train)*100;

acc_train = mean(double(pred_train == y_train)) * 100;
fprintf('\nTraining Set Accuracy: %f\n', acc_train);
toc;

%% Classifier validation

tic;
fprintf('\nPredicting on validation set\n');
[pred_valid,c_valid,cp_valid] = predict(Theta1,Theta2,X_valid);

%Rest efficiency
TP_rest_valid = length(find(pred_valid(find(y_valid==1))==1));
FP_rest_valid = length(find(pred_valid(find(y_valid~=1))==1));
TN_rest_valid = length(find(pred_valid(find(y_valid~=1))~=1));
FN_rest_valid = length(find(pred_valid(find(y_valid==1))~=1));

Se_rest_valid = TP_rest_valid/(TP_rest_valid+FN_rest_valid)*100;
PPV_rest_valid = TP_rest_valid/(TP_rest_valid+FP_rest_valid)*100;
FPR_rest_valid = FP_rest_valid/(FP_rest_valid+TN_rest_valid)*100;


%Left hand motor imagery efficiency
TP_left_valid = length(find(pred_valid(find(y_valid==2))==2));
FP_left_valid = length(find(pred_valid(find(y_valid~=2))==2));
TN_left_valid = length(find(pred_valid(find(y_valid~=2))~=2));
FN_left_valid = length(find(pred_valid(find(y_valid==2))~=2));

Se_left_valid = TP_left_valid/(TP_left_valid+FN_left_valid)*100;
PPV_left_valid = TP_left_valid/(TP_left_valid+FP_left_valid)*100;
FPR_left_valid = FP_left_valid/(FP_left_valid+TN_left_valid)*100;


%Right hand motor imagery efficiency
TP_right_valid = length(find(pred_valid(find(y_valid==3))==3));
FP_right_valid = length(find(pred_valid(find(y_valid~=3))==3));
TN_right_valid = length(find(pred_valid(find(y_valid~=3))~=3));
FN_right_valid = length(find(pred_valid(find(y_valid==3))~=3));

Se_right_valid = TP_right_valid/(TP_right_valid+FN_right_valid)*100;
PPV_right_valid = TP_right_valid/(TP_right_valid+FP_right_valid)*100;
FPR_right_valid = FP_right_valid/(FP_right_valid+TN_right_valid)*100;


acc_valid = mean(double(pred_valid == y_valid)) * 100;
fprintf('\nCross-Validation Set Accuracy: %f\n',acc_valid);
toc;

fprintf('\n    Set    | Accuracy |   Rest: Se/+P/FPR   |  Left: Se/+P/FPR  |  Right: Se/+P/FPR');
fprintf('\n Training  |  %2.2f   |  %2.2f/%2.2f/%2.2f  |  %2.2f/%2.2f/%2.2f  |  %2.2f/%2.2f/%2.2f',acc_train, Se_rest_train, PPV_rest_train,FPR_rest_train,Se_left_train, PPV_left_train,FPR_left_train,Se_right_train, PPV_right_train,FPR_right_train);
fprintf('\nValidation |  %2.2f   |  %2.2f/%2.2f/%2.2f  |  %2.2f/%2.2f/%2.2f  |  %2.2f/%2.2f/%2.2f\n',acc_valid, Se_rest_valid, PPV_rest_valid,FPR_rest_valid,Se_left_valid, PPV_left_valid,FPR_left_valid,Se_right_valid, PPV_right_valid,FPR_right_valid);

fprintf('Exporting results to excel file...\n');
Excel_export_matrix1 = [parameter_Matrix, acc_train, acc_valid, Se_rest_train, PPV_rest_train,FPR_rest_train,Se_left_train,PPV_left_train,FPR_left_train,Se_right_train,PPV_right_train, FPR_right_train, Se_rest_valid, PPV_rest_valid,FPR_rest_valid,Se_left_valid, PPV_left_valid,FPR_left_valid,Se_right_valid, PPV_right_valid,FPR_right_valid];
%xlswrite('results_BCI(1).xlsx',Excel_export_matrix1);

%% Classifier testing

% fprintf('Predicting on test set');
% pred_valid = predict(Theta1,Theta2,X_test);
% fprintf('\nCross-Validation Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);

toc;
