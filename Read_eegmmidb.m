%% Read_eegmmidb
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
%     along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

%% Setup variables

% Fcz_index = find(strcmp(header{1}.labels,'Fcz.')); -> Always 4
% C3_index = find(strcmp(header{1}.labels,'C3..'));  -> Always 9
% C4_index = find(strcmp(header{1}.labels,'C4..'));  -> Always 13


% Change folder below to the eegmmidb root folder on your system.

eegmmidb_path = '../eegmmidb';

ext = 'edf'; % Extension of the files to be read;
Tasks = [1,2,3,4,7,8,11,12]; % Tasks related to Motor Imagery of Hands;
electrodes = [4,9,13]; % Electrodes FCz, C3 and C4;
electrodes_to_delete = 1:64;
electrodes_to_delete(electrodes) = []; % All electrodes indexes that we do not want;
data = cell(872,1); % Instantiates data struct; (872 = 8 >Tasks> * 109 subjects)
header = cell(872,1); % Instantiates header struct;

%% Read database

fprintf('\nLoading Database... This should take a couple minutes...');
% try
i = 1;
for record = 1:109 % Read all records
    fprintf('\n%d%%',floor(record/(109/100)));
    folder = [eegmmidb_path '/S' sprintf('%03d',record)]; 
    D = dir([folder '/*.' ext]);
    D = D(Tasks); % Read only from tasks related to right and left hand M.I.
    for task = 1:length(D); % Read all tasks specified by <Tasks>
        [data{i},header{i}] = ReadEDF([folder '/' D(task).name]);
        data{i}(electrodes_to_delete) = []; % Delete all leads that are not FCz, C3 or C4 (your RAM thanks you!)
        i=i+1;
    end
end
fprintf('\n100%%\nDatabase loaded successfully!\n');
clear eegmmidb_path i D electrodes electrodes_to_delete ext folder record task Tasks
% catch
%     clear eegmmidb_path i D electrodes electrodes_to_delete ext folder record task Tasks
%     error('Make sure you have enough free RAM (at least 1GB) and the path is set correctly');
% end

