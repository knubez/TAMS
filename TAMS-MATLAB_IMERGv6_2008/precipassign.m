function [systems_classified] = precipassign(systems_classified)
% precipassign
% Created 9/2/2019 by KMNO
% Description:
% ---------------------
% This code assigns rainfall to each of the MCSs identified and classified using IMERG V6 half-hourly 0.1 x0.1 degrees (10km x 10km aprrox.) precipitation (rainfall rates) field.
% Field from combined microwave-IR estimate withgauge calibration

% Domain
% First paper
% xlims =[-35 45];
% ylims=[0, 30];
% Revised paper
xlims = [-45 60];
ylims = [-5  30];

for k = 1:length(systems_classified)
    systems=systems_classified(k).System;
    
    if strcmp(systems_classified(k).Class,'DSL')==0 % do not assign rain to disorganized systems
        for l=1:length(systems)
            
            % IMERG Data location
            folder='/gpfs/group/jle7/default/kmn18/graduateresearch/NASA_IMERG_PRECIP_AUG_SEPT_2008/';
            
            Year=systems(l).year;
            Month=systems(l).month;
            Day=systems(l).day;
            yearmonthday=strcat(Year,Month,Day);
            Hour=(systems(l).hour);
            yearmonthday=strcat(Year,Month,Day);
            
            list=dir([folder '3B-HHR.MS.MRG.3IMERG.' yearmonthday '-S' Hour '0000' '*.nc4']);
            filepath=[folder list.name];
            ncid=netcdf.open(filepath,'nc_nowrite');
            datafile =list.name;
            [~, filebasename, extension] = fileparts(datafile);
           
            
            varidp=netcdf.inqVarID(ncid,'precipitationCal'); % combined microwave-IR estimate with gauge calibration [mm /hr]
            varidlat=netcdf.inqVarID(ncid,'lat'); % 353
            varidlon=netcdf.inqVarID(ncid,'lon'); % 1175  [degrees east] -180 180
            
            precip= double(netcdf.getVar(ncid,varidp));
            precip(precip==0)=NaN; % get rid of 0 values
            
            latitude=double(netcdf.getVar(ncid,varidlat));
            longitude=double(netcdf.getVar(ncid,varidlon));
            
            % First define a meshgrid with the resolution desired ( EUMETSAT ~3KM (0.0degrees) over the large static doamin in study
            delta = 0.035; % EUMETSAT ~ 3km
            targetLats =ylims(1):delta:ylims(2); targetLons = xlims(1):delta:xlims(2);
            [lonArray,latArray]=meshgrid(targetLons,targetLats);
            
            % Find large static domain in study from precip data
            [n]=find(latitude>=ylims(1) & latitude<=ylims(2));
            [m]=find(longitude>=xlims(1) & longitude<=xlims(2));
            lat=latitude(n);
            lon=longitude(m);
            Precip=precip(n,m);
            
            % Interpolate Precip datat to higher resolution grid
            vq=interp2(lon,lat,Precip,lonArray,latArray,'linear');
            
            % Use points inside CE
            
            [in,on]=inpolygon(lonArray,latArray,systems(l).x,systems(l).y); % current cloud edge information
            
            pointsincloud=find(in>0); % using the mask "in" obtain the points inside the CE
            
         
            rain_rate = vq(pointsincloud); %pointsincloud); % points of rain_rate within the CE, zeros are marked as NaNs;
      
            rain_rate(rain_rate<=0.4)= NaN; %trace (0.2mm)*2 % Acount for trace values [ mm / hr]
            
            
            rain_rate(isnan(rain_rate))=[];
            h=size(rain_rate); % whcih is same size as rain_acc 
            % Save data
            systems_classified(k).Raindata(l).rain_rate= rain_rate; % Evrything less or equal to zero is NaN
           
            
            systems_classified(k).Raindata(l).rain_acc= rain_rate*(1/2); % [mm] multiply by 1/2 since data is every half hour to get rain accumulation, % Evrything less or equal to zero is NaN
           
        
            x= vq.*in;
            %x(x==0)=NaN;
            
            x(x<=0.4)=NaN;% includes both outside of CE and trace
            
            % Note: ran_rate_2, is the variable saved with rows and columns info for plottitng purposes but  calculations are done with 1-D rain_rate variables although numbers are same in both types of calculations  
            % x(isnan(x))=[]; this can be done and both area_ave_rain_rate calculations will be the same. But as stated above, we preserve the rows and columns on this variable for plotting purposes.
            systems_classified(k).Raindata(l).rain_rate_2=x;
            x(isnan(x))=[];
            hs= size(x);
            systems_classified(k).Raindata(l).area_ave_rain_rate_2 = mean(mean(x))/(hs(1)*hs(2));
            
            
           
            rain_acc_2 = x*(1/2);
            systems_classified(k).Raindata(l).area_sum_rain_acc_2 = sum(sum(rain_acc_2))/(hs(1)*hs(2));
          
            % Stats
            rain_acc = rain_rate*(1/2);
            systems_classified(k).Raindata(l).area_sum_rain_acc = sum(rain_acc)/(h(1)*h(2));
            % using nan as points with zero precip were defined Nans as well as points of less than 0.2 value
            
            systems_classified(k).Raindata(l).area_ave_rain_rate = mean(rain_rate)/(h(1)*h(2));
            % using nan as points with zero precip were defined Nans as well as points of less than 0.2 value
            
            systems_classified(k).Raindata(l).indices_in=in; 
            
            % Save string of precip file name
            systems_classified(k).Raindata(l).filebasename = filebasename;
            
        end
    end
    
end
% Extra 
%new_lat = latArray.* in;
%systems_classified(k).Raindata(l).rain_lat=new_lat; % latArray(pointsincloud);

%new_lon = lonArray.* in;
%systems_classified(k).Raindata(l).rain_lon=new_lon ; % lonArray(pointsincloud);

disp('precipitation assigment is done')
end
 
