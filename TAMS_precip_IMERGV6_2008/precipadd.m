function [systems_classified] = precipadd(systems_classified)
% PRECIPADD
%DESCRIP
%----------------------------
% This function assigns precipitation to every one of the systems
% classified
% Regridding and interpolating for differences in hours

%MOdified 1/3/2018 -- Kelly Nunez Ocasio
%% Adding path for precip
%path of all rainfall data
databasedir_precip='/gpfs/group/jle7/default/kmn18/TRMM/Version7/';
addpath('/gpfs/group/jle7/default/kmn18/graduateresearch/matlabcodes/borders'); % add borders
% Create new structure for rainfall tha will saved next to classified systems
tic;
%Raindata=struct();
%Large domain
%xlims = [-8,8];
%ylims = [12,17];
% 
% xlims = [-76,60];
% ylims = [-5,50];
xlims =[-35 45];
ylims=[0, 30];
% xlims=[-40,35];
% ylims=[-5,23];
for i =1:length(systems_classified)
   % i
    
    Rainy=struct();
    systems=systems_classified(i).System;
    if strcmp(systems_classified(i).Class,'DSL')==0 % do not assign rain to disorganized systems
        dummy=0;
        for j=1:length(systems)-1
           % j
            Hour=str2double(systems(j).hour); 
            trmmhours=[0,03,06,09,12,15,18,21];
            inside=ismember(Hour,trmmhours);% checking of ellipsoid has a designated TRMM hour
            if inside==0 % If ellipsoid does not have coinciding TRMM data then take the TRMM data in
                % between the current ellipsoid hour and the next
                if Hour==str2double(systems(j+1).hour)
                    b={systems_classified(i).System.hour};
                    bb=b(j:length(b));
                    bb_unique=unique(bb);
                    if length(bb_unique)==1
                        Hour_next=str2double(bb_unique(1));
                    else
                        Hour_next=str2double(bb_unique(2));
                    end
                else
                    
                    Hour_next=str2double(systems(j+1).hour);
                end
                diff_hour=median([Hour Hour_next]);%(Hour_next+Hour)/2; % taking the value in between
                
                inside_2=ismember(diff_hour,trmmhours);% checking ifhour in between curretn ellipsoid and nex is inside TRMM
                if inside_2==1  % If hour in between of current ellipsoid and next is inside TRMM obtain that rainfall data and
                    %interpolate ellipsoids
                    % Following steps are done on order to find rainfall data pertinent to the date of the satellite data in a way it can be done for
                    % longer period of time without having to specify  months or days
                    % dummy=dummy+1;
                    Year=systems(j).year;
                    Month=systems(j).month;
                    Day=systems(j).day;
                    yearmonthday=strcat(Year,Month,Day);
                    Month=str2num(systems(j).month);
                    Monthfordir=[{'January'},{'February'},{'March'},{'April'},{'May'},{'June'},{'July'},{'August'}, ...
                        {'September'},{'October'},{'November'},{'December'}];
                    monthy=cell2mat(Monthfordir(Month));
                    % Here use hour of TRMM which will be diff_hour
                    if diff_hour<10
                        systems_precip=[databasedir_precip,Year,'/',monthy,'_', Year,'/3B42.',yearmonthday,'.','0',num2str(diff_hour),'.7A.nc'] ;
                    else
                        systems_precip=[databasedir_precip,Year,'/',monthy,'_', Year,'/3B42.',yearmonthday,'.',num2str(diff_hour),'.7A.nc'] ;
                    end
                    
                    %% Obtaining netcdf info of TRMM rainfall data
                    % Open netCDF file
                    ncid = netcdf.open(systems_precip,'nc_nowrite');
                    
                    % Get information about the contents of the file
                    [ndims,nvars,ngatts,unlimdimid] = netcdf.inq(ncid);
                    %inqVar(ncid)-returns information about variable %ndims-number of dimensions %
                    
                    % Get information about each variable in the file
                    for v = 1:nvars
                        [varnam,xtype,dimids,numatts] = netcdf.inqVar(ncid,v-1);
                    end
                    
                    % Get values for each variable
                    data.precipitation = netcdf.getVar(ncid,3);
                    % Set everything zero (no rain ) to NaN
                    data.precipitation(data.precipitation==0)=NaN;
                    data.relativeError = netcdf.getVar(ncid,4);
                    data.lat = netcdf.getVar(ncid,2);
                    data.lon = netcdf.getVar(ncid,1);
                    
                    %                     xlims = [-8,8];
                    %                     ylims = [12,17];
                    %%%%% For plotting original plot purposes%%%%%%%%%%%
                    % Return number of precip values in each row and column
                    [preciprow,precipcol] = size(data.precipitation);
                    
                    % Create 1440*400 latitude and longitude matrixes because lat and lon are not smae sizes as precip
                    lats = repmat(data.lat',preciprow,1);
                    lons = repmat(data.lon,1,precipcol);
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    % ************** from Young and Kelly  *******
                    % delta = 0.5; targetLats =-5:delta:23; targetLons = -40:delta:25;
                    delta = 0.035; targetLats =ylims(1):delta:ylims(2); targetLons = xlims(1):delta:xlims(2);
                    
                    
                    % 3- meshgrid TRMM
                    [lonArray,latArray]=meshgrid(targetLons,targetLats);
                    %                 % 4- Nearest neighbor to interp
                    
                    
                    % Using scatter interpolant
                    [n]=find(data.lat>=ylims(1) & data.lat<=ylims(2));
                    [m]=find(data.lon>=xlims(1) & data.lon<=xlims(2));
                    lat=data.lat(n);
                    lon=data.lon(m);
                    Precip=data.precipitation(m,n);
                    
                    %  vq=interp2(data.lat,data.lon,data.precipitation,latArray,lonArray);
                    vq=interp2(lat,lon,Precip,latArray,lonArray); % MAKE SURE TO SWITCH X AND Y WHEN USING INTERP2 % I had originals here data.pre...
                    
                    
                    % 5- Use insode poly and save pointd inside poly
                    %%%%%%%%%%%%%%
                    % Use equation to interpolate ellipsoids
                    B=(diff_hour-Hour)/(Hour_next-Hour);
                    %lengthy=length(Raindata)+1;
                    systems_classified(i).Raindata(j).Xe=B*(systems(j+1).xe) + (1-B)*systems(j).xe;
                    systems_classified(i).Raindata(j).Ye=B*(systems(j+1).ye) + (1-B)*systems(j).ye;
                    systems_classified(i).Raindata(j).ZTX=B*(systems(j+1).ztx) + (1-B)*systems(j).ztx;
                    systems_classified(i).Raindata(j).ZTY=B*(systems(j+1).zty) + (1-B)*systems(j).zty;
                    systems_classified(i).Raindata(j).BT=B*(systems(j+1).bt) + (1-B)*systems(j).bt;
                    systems_classified(i).Raindata(j).AT=B*(systems(j+1).at) + (1-B)*systems(j).at;
                    systems_classified(i).Raindata(j).ALPHAT=B*(systems(j+1).alphat) + (1-B)*systems(j).alphat;
                    %%%%%%%%%%%%%%%%%%
                    [in,on]=inpolygon(lonArray,latArray,systems_classified(i).Raindata(j).Xe,systems_classified(i).Raindata(j).Ye);
                    
                    pointsinellipse=find(in>0); % using the mask "in" obtain the points inside the ellipse
%                     disp('sizep of point in ellipse')
%                     size(pointsinellipse)
                    raindatain=in.* vq; % this can be either points that are raining or not
                    pointsinrain= find(raindatain>0.2); % at this point it will save the points inside the ellipsoid greater than 0.2
                    
                    frac_rain_area= (length(pointsinrain)/length(pointsinellipse))*100; % percentage terms
                    
                    % Rain data
                    raindatain(raindatain<=0.2)=NaN;  % <0.2
                    % Only save the points the points that are not trace or NaNs thus, pointsinraining >0.2
                                       
                    %systems_classified(i).Raindata(j).rainratein=raindatain;
                    systems_classified(i).Raindata(j).rainratein=raindatain(pointsinrain); %new
                    % If you multiply raindata in by 3 you get rain % accumulaiton [ mm ] but from WMO trace is everything below 0.2
                    %raindatain(raindatain<=0)=NaN;  % <0.2
                   % Save the Average Rainrate of Precipitating area
                    systems_classified(i).Raindata(j).ave_rainrate_area=mean(raindatain(pointsinrain));
                    
                    rainaccin=raindatain.*3;
                    %systems_classified(i).Raindata(j).rainaccin=rainaccin;
                    systems_classified(i).Raindata(j).rainaccin=rainaccin(pointsinrain); %new 
                     % Save the Average Rain accumulaiton  of Precipitating area
                    systems_classified(i).Raindata(j).ave_rainacc_area=mean(rainaccin(pointsinrain));
                     
                    systems_classified(i).Raindata(j).rainlat=latArray(pointsinrain);
                    systems_classified(i).Raindata(j).rainlon=lonArray(pointsinrain);
                    systems_classified(i).Raindata(j).rainfrac=round(frac_rain_area);
                    systems_classified(i).Raindata(j).pointinelliplen=length(pointsinellipse);
                    

                    % Save date and time information
                    systems_classified(i).Raindata(j).Year=systems(j).year;
                    systems_classified(i).Raindata(j).Month=systems(j).month;
                    systems_classified(i).Raindata(j).Day=systems(j).day;
                    systems_classified(i).Raindata(j).Hour=num2str(diff_hour); % TRMM hour
%                     disp('hour')
%                     num2str(diff_hour);
                    systems_classified(i).Raindata(j).Minutes=systems(j).minutes;
                    systems_classified(i).Raindata(j).Seconds=systems(j).seconds;
                    year=str2double(systems(j).year);
                    month=str2double(systems(j).month);
                    day=str2double(systems(j).day);
                    hour=diff_hour;
                    minutes=str2double(systems(j).minutes);
                    seconds=str2double(systems(j).seconds);
                    time=datetime(year,month,day,hour,minutes,seconds);
                    str=datestr(time);
                    systems_classified(i).Raindata(j).strindate=str;
                    
                end
                
            else % if ellipsoid has matching TRMM data
                %disp ('KELLYYYYYY')
                Year=systems(j).year;
                Month=systems(j).month;
                Day=systems(j).day;
                yearmonthday=strcat(Year,Month,Day);
                Month=str2num(systems(j).month);
                Monthfordir=[{'January'},{'February'},{'March'},{'April'},{'May'},{'June'},{'July'},{'August'}, ...
                    {'September'},{'October'},{'November'},{'December'}];
                monthy=cell2mat(Monthfordir(Month));
                % Here use hour of TRMM which will be diff_hour
                if Hour<10
                    systems_precip=[databasedir_precip,Year,'/',monthy,'_', Year,'/3B42.',yearmonthday,'.','0',num2str(Hour),'.7A.nc'] ;
                else
                    systems_precip=[databasedir_precip,Year,'/',monthy,'_', Year,'/3B42.',yearmonthday,'.',num2str(Hour),'.7A.nc'] ;
                end
                
                %% Obtaining netcdf info of TRMM rainfall data
                % Open netCDF file
                ncid = netcdf.open(systems_precip,'nc_nowrite');
                
                % Get information about the contents of the file
                [ndims,nvars,ngatts,unlimdimid] = netcdf.inq(ncid); %inqVar(ncid)-returns information about
                %variable %ndims-number of dimensions %
                
                % Get information about each variable in the file
                for v = 1:nvars
                    [varnam,xtype,dimids,numatts] = netcdf.inqVar(ncid,v-1);
                end
                
                % Get values for each variable
                data.precipitation = netcdf.getVar(ncid,3);
                % Set everything zero (no rain ) to NaN
                data.precipitation(data.precipitation==0)=NaN;
                
                data.relativeError = netcdf.getVar(ncid,4);
                data.lat = netcdf.getVar(ncid,2);
                data.lon = netcdf.getVar(ncid,1);
                
                
                %%%%% For plotting original plot purposes%%%%%%%%%%%
                % Return number of precip values in each row and column
                [preciprow,precipcol] = size(data.precipitation);
                
                % Create 1440*400 latitude and longitude matrixes because lat and lon ar enot smae sizes as precip
                lats = repmat(data.lat',preciprow,1);
                lons = repmat(data.lon,1,precipcol);
                
                
                % delta = 0.5; targetLats =-5:delta:23; targetLons = -40:delta:25; % bigger domain
                delta = 0.035; targetLats =ylims(1):delta:ylims(2); targetLons = xlims(1):delta:xlims(2); %.06
                %delta = 0.0; targetLats =12:delta:17; targetLons = -8:delta:8;
                
                % 3- meshgrid TRMM
                [lonArray,latArray]=meshgrid(targetLons,targetLats);
                % interp
                
                %% Using scatter interpolant
                [n]=find(data.lat>=ylims(1) & data.lat<=ylims(2));
                [m]=find(data.lon>=xlims(1) & data.lon<=xlims(2));
                lat=data.lat(n);
                lon=data.lon(m);
                Precip=data.precipitation(m,n);
                
                %  vq=interp2(data.lat,data.lon,data.precipitation,latArray,lonArray);
                vq=interp2(lat,lon,Precip,latArray,lonArray); % MAKE SURE TO SWITCH X AND Y WHEN USING INTERP2 % I had originals here data.pre...
                
                % 5- Use insode poly and save pointd inside poly
                
                [in,on]=inpolygon(lonArray,latArray,systems(j).xe,systems(j).ye); % current ellipsoid information
                
                %inon=in | on;
                pointsinellipse=find(in>0); % using the mask "in" obtain the points inside the ellipse
%                  disp('sizep of point in ellipse')
%                     size(pointsinellipse)
                raindatain=in.* vq; % this can be either points that are raining or not
                pointsinrain= find(raindatain>0.2); % at this point it will save the points inside the ellipsoid greater than 0.2 and thus raining
                
                frac_rain_area= (length(pointsinrain)/length(pointsinellipse))*100; % percentage terms
                
   
                
               % lengthy=length(Raindata)+1;
                % Save ellipsoid information 
                systems_classified(i).Raindata(j).Xe=systems(j).xe;
                systems_classified(i).Raindata(j).Ye=systems(j).ye;
                systems_classified(i).Raindata(j).ZTX=systems(j).ztx;
                systems_classified(i).Raindata(j).ZTY=systems(j).zty;
                systems_classified(i).Raindata(j).BT=systems(j).bt;
                systems_classified(i).Raindata(j).AT=systems(j).at;
                systems_classified(i).Raindata(j).ALPHAT=systems(j).alphat;
                
                % Save rain information
                raindatain(raindatain<=0.2)=NaN;  % <0.2
                %systems_classified(i).Raindata(j).rainratein=raindatain;  % unit of [mm/ hr]
                systems_classified(i).Raindata(j).rainratein=raindatain(pointsinrain); %new % unit of [mm/ hr]
                % If you multiply raindata in by 3 you get rain % accumulaiton [ mm ] but from WMO trace is everything below 0.2
                
                %raindatain(raindatain<=0)=NaN;  % <0.2
                %Save the Average Rainrate of Precipitating Area
%                 disp('ave_rainrate')
%                 mean(raindatain(pointsinrain))
                systems_classified(i).Raindata(j).ave_rainrate_area=mean(raindatain(pointsinrain));
                rainaccin=raindatain.*3;
                %systems_classified(i).Raindata(j).rainaccin=rainaccin;
%                 disp('ave_rainacc')
%                 mean(rainaccin(pointsinrain))
%                 disp('frac')
%                 round(frac_rain_area)
%                 disp('pointinrain')
%                 length(pointsinrain)
                systems_classified(i).Raindata(j).rainaccin=rainaccin(pointsinrain); %new
                %Save the Average Rain accumulation of Precipitating area
                systems_classified(i).Raindata(j).ave_rainacc_area=mean(rainaccin(pointsinrain));
                systems_classified(i).Raindata(j).rainlat=latArray(pointsinrain);
                systems_classified(i).Raindata(j).rainlon=lonArray(pointsinrain);
                systems_classified(i).Raindata(j).rainfrac=round(frac_rain_area);
                systems_classified(i).Raindata(j).pointinelliplen=length(pointsinellipse);
                
                % Save date and time information
%                 disp('hour')
%                 systems(j).hour
                systems_classified(i).Raindata(j).Year=systems(j).year;
                systems_classified(i).Raindata(j).Month=systems(j).month;
                systems_classified(i).Raindata(j).Day=systems(j).day;
                systems_classified(i).Raindata(j).Hour=(systems(j).hour);
                systems_classified(i).Raindata(j).Minutes=systems(j).minutes;
                systems_classified(i).Raindata(j).Seconds=systems(j).seconds;
                year=str2double(systems(j).year);
                month=str2double(systems(j).month);
                day=str2double(systems(j).day);
                hour=str2double(systems(j).hour);
                minutes=str2double(systems(j).minutes);
                seconds=str2double(systems(j).seconds);
                time=datetime(year,month,day,hour,minutes,seconds);
                str=datestr(time);
                systems_classified(i).Raindata(j).strindate=str;
                
                
            end
        end
        lengthy=0;
        %systems_classified(i).Raind=Raindata;
    end

    lengthy=0;
    
    
    
    
end

disp('precip is done')
toc;
end




