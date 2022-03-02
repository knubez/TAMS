% --------------------------------------------------------------------------------------------------
% Identification of systems
% This code open MSG SEVIRI VIS Channel data and plot the Brightness
% Temperature a.k.a, Tb, ( ch9 after converted from counts->radiance->temperature) .
% It plots Tb  with respect to location.
%
%
% DESCRIP
% 1. This code open netcdf file for each hour under respective folders of
%    each month.
% 2. It calls funciton identify.m to create image and develop calculations
%    of areas of interest.
% 3. Saves all calculations in mat file: "date_219and235.mat" with variable
%    name "data_day" for each day with respective month day and year info. 

% Note: Can run infinite files as long as they are in order and in a same
% folder  
%
%
% HIST
% Created JUly 27Th 2017 by Kelly M. Nunez Ocasio March 6th-  
% modified 

% --------------------------------------------------------------------------------------------------
 
tic;
  
% -----------------------------------
% locations of files 
addpath('/gpfs/group/jle7/default/kmn18/graduateresearch/matlabcodes/altmany-export_fig-2763b78'); % add export_fig

addpath('/gpfs/group/jle7/default/kmn18/graduateresearch/matlabcodes/borders'); % add borders

%databasedir ='/gpfs/group/jle7/default/kmn18/graduateresearch/MSG_IR_Aug_Sept_2006/archive.eumetsat.int/umarf/onlinedownload/KellyNunez/1258484/1/';
databasedir ='/gpfs/group/jle7/default/kmn18/graduateresearch/Test/';
%databasedir='/gpfs/group/jle7/default/kmn18/graduateresearch/MSG_Infrared_Sept_11_12_2006/archive.eumetsat.int/umarf/onlinedownload/KellyNunez/1233816/';
%outputbasedir ='/gpfs/group/jle7/default/kmn18/graduateresearch/outputfiles/Aug_Sept2006MSGdailyfileslargedomain/'; % location of outputfiles 
outputbasedir ='/gpfs/group/jle7/default/kmn18/graduateresearch/outputfiles/'; % location of outputfiles 
% Which files to search for in folders
myfiles=dir([databasedir, '*EUMG_20060801*.nc' ]);    
kiddy=1;
previous_day = 0;
for j = 1:numel(myfiles) % every 2 hours     
    % Set name of current file to a variable ex. 3B42.2...  
    datafile = myfiles(j).name;
    [~, filebasename, extension] = fileparts(datafile);
    
    %Adding the data file to the data path % concatenate folder name with specific data name
    filetoload = strcat(databasedir, datafile); 
        
    % thing to save for each day
    if j == 1  % first file
        data_day = struct([]); % start a structure to save info of day
        i = 1; % counts hour 1
        % esle if you are in a different day it saves and closes previous data structure and opens a new one
    elseif str2double(filebasename(65:66)) ~= previous_day
        
        % save MAT file for the previous day
        time_ID = [data_day(1).year data_day(1).month data_day(1).day];
        matfilepath = [outputbasedir time_ID '_219and235.mat'];
        save(matfilepath, 'data_day','-v7.3')
        
        % start structure for new day
        clear data219 data235 data_day
        data_day = struct([]);
        i = 1;
        
    else  % then if you are in the same day keep adding hours and thus i=i+1;
        clear data219 data235
        i = i + 1;
        
    end

    % log where we are
    fprintf('\n\n j = %d, i = %d \n\n\n', j, i)

  

    % do the contouring and classify and such
    [data219, data235,data_day] = identify(kiddy, i,data_day,filetoload,filebasename); 
    data_day(i).data219struct = data219;
    data_day(i).data235struct = data235;
    kiddy=kiddy+1;    
previous_day =str2double(data_day(i).day); % each time previous day will be updated to the current file 
end
toc;
