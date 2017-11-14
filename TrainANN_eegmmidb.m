%% TrainANN_eegmmidb
% <Description>
% <Functionality>

%  Copyright (C) 2016  Vinicius Queiroz
% 
%     This file is part of BCI-UFOP.
% 
%     BCI-UFOP is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     BCI-UFOP is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with BCI-UFOP.  If not, see <http://www.gnu.org/licenses/>.

%% Declaration of Variables
input_layer_size  = size(X_train,2);  % Feature Vector Input
m = size(X_train,1); % Number of examples

%% Finding Theta values
if load_w == 0
    
    num_labels = 3;            % Number of labels  
    % Randomly initializes the parameters
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

    % Performs feedforward and backprogation iteratively
    options = optimset('MaxIter', max_iter);
    costFunction = @(p) nnCostFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, X_train, y_train, lambda);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

    % Reshape the parameters optimized
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));

    % Save the parameters
    fprintf('Saving weights of ANN to file......');

    save ('weights2.mat','Theta1','Theta2');
else
    fprintf('Loading saved weights of ANN from file......');
    load('weights.mat');
end