function [ systems_classified, Systemss, Positions ] = savingfiles(systems_classified,first_date,last_date,outputbasedir)
%savingfiles-name of function
%----------------
% 1. This function saves the structures in order to accleerate the process
% of the code
% 2.

% HIST
% Created March  24th 2018 by Kelly M. Nunez Ocasio
% --------------------------------------------------------------------------------------------------------------------
% Here I make structure with just the rain info
Rain=struct();
Rain=systems_classified;
Rain=rmfield(Rain,'System');
% Here I make structure with time, positions and classification of each system classified even if DSL (all info)
Systemss=struct();
Systemss=systems_classified;
Systemss=rmfield(Systemss,'Raindata');

%% Save big  mat file for all period in study without raindata
time_ID=[first_date last_date];
matfilepath = [outputbasedir time_ID 'MCStracked_10ms.mat'];
save(matfilepath, 'Systemss','-v7.3')

%% Save matfile for all period for Rain data
matfilepath = [outputbasedir time_ID 'MCSrain_10ms.mat'];
save(matfilepath, 'Rain','-v7.3')
%Note: This part should be added at the end of overlap_with_precip_new_latest.m to have a smaller matfile for MCSs to be matched to hurricanes
for k=1:length(Systemss)
    Positions(k).Class=Systemss(k).Class;
    Positions(k).maxareatime=Systemss(k).maxareatime;
    Positions(k).maxstd219time=Systemss(k).maxstd219time;
    Positions(k).tempmin219time=Systemss(k).tempmin219time;
    Positions(k).eccenmintime=Systemss(k).eccenmintime;
    Positions(k).genesistime=Systemss(k).genesistime;
    Positions(k).terminationtime=Systemss(k).terminationtime;
    Positions(k).max_area_ave_rain_rate_time=Systemss(k).max_area_ave_rain_rate_time;
    Positions(k).max_area_ave_rain_rate_time=Systemss(k).max_area_ave_rain_rate_time;

    %Save duration of system
    % if strcmp(Systemss(k).Class,'DSL')==0
    initial_time=Systemss(k).System(1).datestr;
    final_date=Systemss(k).System(length([Systemss(k).System])).datestr;
    time_duration=hours(final_date-initial_time);
    Positions(k).reldura=time_duration;
    %Save speed of systems by taking distamnce between last and first
    %position and multilyginng by duration (hours)
    disty=distance(Systemss(k).System(1).lat,Systemss(k).System(1).lon,Systemss(k).System(length([Systemss(k).System]) ...
        ).lat,Systemss(k).System(length([Systemss(k).System])).lon)*(pi/180)*6371;
    velocity=(disty/Positions(k).reldura)* (5/18); % from km/hr to m/s with 5/18
    Positions(k).velocity=velocity;
    
    for l=1:length(Systemss(k).System)
        yearnow =Systemss(k).System(l).year ;
        monthnow  =Systemss(k).System(l).month ;
        daynow =Systemss(k).System(l).day;
        hournow =Systemss(k).System(l).hour;
        minutesnow=Systemss(k).System(l).minutes;
        secondsnow  =Systemss(k).System(l).seconds;
        strinnow=[yearnow, '-', monthnow, '-', daynow, ' ', hournow, ':', minutesnow, ':' secondsnow];
        timenow=datetime(strinnow,'InputFormat','yyyy-MM-dd HH:mm:ss');
        Positions(k).system(l).date=timenow;
        Positions(k).system(l).lat=Systemss(k).System(l).lat;
        Positions(k).system(l).lon=Systemss(k).System(l).lon;
        Positions(k).system(l).lat_dist=Systemss(k).System(l).lat_dist;
        Positions(k).system(l).lon_dist=Systemss(k).System(l).lon_dist;
        
        % ADD x and y and thus cloud edges
        Positions(k).system(l).x=Systemss(k).System(l).x;
        Positions(k).system(l).y=Systemss(k).System(l).y;
        
        % ADD data219struct
        Positions(k).system(l).data219struct=Systemss(k).System(l).data219struct;
        if strcmp(Systemss(k).Class,'DSL')==0
            Positions(k).system(l).localkids=Systemss(k).System(l).localkids;
            Positions(k).system(l).localparent=Systemss(k).System(l).localparent;
        end
    end
    % end
end
%% matfile for all period for Positions data and Frequency info
matfilepath = [outputbasedir time_ID 'MCSpositions_10ms.mat'];
save(matfilepath, 'Positions','-v7.3')
disp('time code finishes saving all matfiles ')
toc;

end
