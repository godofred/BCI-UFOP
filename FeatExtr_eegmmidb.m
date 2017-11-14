%% FeatExtr_eegmmidb
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




%% Generates a regular graph, with the weights corresponding to the Euclidian Distance between all points

window_number_of_samples = length(window.FCz);

W = zeros(window_number_of_samples); % <W> is the weight matrix of the graph;

for i=1:window_number_of_samples
    for j=1:window_number_of_samples
            W(i,j) = sqrt((window.FCz(i)-window.FCz(j))^2 + (window.C3(i)-window.C3(j))^2 + (window.C4(i)-window.C4(j))^2 + (window.t(i)-window.t(j))^2); % Euclidian Distance
    end
end

biggest_weight = max(W(:));
W = W/biggest_weight; %% Normalizes between 0 and 1;

dv = zeros([1 2*m]);
pv = zeros([1 3*m]);


%% Dynamic Evolution

for i = 1:m
    A = ones(size(W)); % <A> is the adjacency matrix of the graph (0 means not connected, 1 means connected)
    A(W>=t(i)) = 0; % Disconnects all vertices that have a distance biggest than specified by the threshold <t>

    % Connectivity Degree features
    k = zeros([1 window_number_of_samples]);
    for j=1:window_number_of_samples
        k(j) = sum(A(j,:)); % normalizes
    end

    dv((2*i)-1)=mean(k);
    dv((2*i))=max(k);

    % Probabilistic features
    
    H = 0;
    E = 0;
    P = 0;
    
    degrees = sum(A);
    [r, c] = size(A);
    ndegrees = c;
    
    matrix = zeros(ndegrees);
    
    for i2=1:r
        for j2=1:c
            if(A(i2,j2) == 1)
                matrix(degrees(i2),degrees(j2)) = matrix(degrees(i2),degrees(j2))+1;
            end        
        end        
    end
    
    
    nedges = sum(A(:));
    matrix = matrix/nedges;
    
    number_degrees  = length(degrees);

    for i3=1:number_degrees
       pk = matrix(degrees(i3),degrees(i3));
       H = H+pk*log2(pk+eps);
       E = E + pk^2;
       P = P + pk;        
    end
    H = -H;
    P = P/number_degrees;
   
    pv((3*i)-2)= H;
    pv((3*i)-1)= E;
    pv((3*i))  = P;
    

    
    
        
end


if strcmp(header{task}.annotation.event(event), 'T0')
    label = 1;
elseif strcmp(header{task}.annotation.event(event), 'T1')
    label = 2;
elseif strcmp(header{task}.annotation.event(event),'T2')
    label = 3;
else
    error('Label not existent in task %d, event %d',task,event);
end

featureVector(w,:) = [dv pv];
y(w,1) = label;