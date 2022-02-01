    % --------------------------------------------------------------------------------------------------
% Runnin overlapping test 
% This code uses the .mat outputfiles from "processing" code to run the
% overlapping test on 235K contours .
%
%  
% 
% DESCRIP
% 1. First it identifies clusters of 219 K that are bigger than 4,000km (to only keep track of the organized systems).
%
% 2. Then it checks if this contours/clusters are inside a 235K 
%
% 3. If there is more than one 219K contour inside a 235K it takes the sum of all the 219K contour areas inside of it, if its >= 4000 km 
%    the 235K contour is selected to be tracked.  
%
% 4. If there is just one 219K contour >= 4000km in area inside a 235K it is also selected for traking.  
%
% 5. Thus, in this process all 235K that do not have 219 areas inside are
%    ignored and eventuall be clasffied as DSL. 
%
% 6. Then overlapping is computed by taking the current contour and comparing it to all the contours of the next hours. If overlap between
%    the overlap region over the smaller contour is >= 50 they are merged and a "kid" is added to that contour where "kid" is ID 
%    of the other contour that matche with.
%
% 7. Tracks are plotted with gif over time and time colored.    
%   
% 
% Note: 
%
% HIST
% Created August 7th 2017 by Kelly M. Nunez Ocasio  
% modified
% Uploading mat files 
clear all;
close all;  
clc ;  
addpath('/gpfs/group/jle7/default/kmn18/graduateresearch/matlabcodes/altmany-export_fig-2763b78'); % add export_fig
addpath('/gpfs/group/jle7/default/kmn18/graduateresearch/matlabcodes/cbrewer/cbrewer');
addpath /gpfs/group/jle7/default/kmn18/graduateresearch/outputfiles

addpath('/gpfs/group/jle7/default/kmn18/graduateresearch/matlabcodes/borders'); % add borders
addpath('/gpfs/group/jle7/default/kmn18/graduateresearch/matlabcodes/geom2d/geom2d'); % add ellipsetopolygon 

%databasedir = '/gpfs/group/jle7/default/kmn18/graduateresearch/outputfiles/'; % location of  case sept 06 mat file
%databasedir = '/gpfs/group/jle7/default/kmn18/graduateresearch/outputfiles/Aug_Sept2006MSGdailyfileslargedomain/'; % location of  Aug_sept 06 large domain
%databasedir = '/gpfs/group/jle7/default/kmn18/graduateresearch/outputfiles/July_Sept2006MSGdailyfileslargedomain/'; %2006 May_Oct
%databasedir = '/gpfs/group/jle7/default/kmn18/graduateresearch/outputfiles/Extra/'; 
%databasedir = '/gpfs/group/jle7/default/kmn18/graduateresearch/outputfiles/test2006/'; %2006 May
databasedir   = '/gpfs/group/jle7/default/kmn18/graduateresearch/outputfiles/2006MSGdailyfileslargedomain_2/';

%outputbasedir = '/gpfs/group/jle7/default/kmn18/graduateresearch/outputfiles/MCStrackedlargedomain_2008_2/'; % location of outputfiles
%outputbasedir = '/gpfs/group/jle7/default/kmn18/graduateresearch/outputfiles/'; % location of outputfiles

tic; 
newsystems=struct();
%newsystems.data219struct=struct();   
ind=1;
count=0; 
  
first_date=[];  
last_date=[]; %'case*.
 
files=dir([databasedir,'2008*.mat']);  %file case study sept 11  
%files=dir([databasedir,'20060911*.mat']);  %file case study sept 11  
  %file case study sept 06
tic; 
for i=1:numel(files) %loop of day
%     clear data_day
%     clear data219
%     clear data235
    load([databasedir, files(i).name]);
    namefile=files(i).name;
    
    % This for naming the matfile
    if i==1
        first_date=namefile(3:9); % (5:12); -- for case 
    elseif i==length(files)
        last_date=namefile(3:8);
    end
    
    [~, filebasename, extension] = fileparts(namefile);
    for j=1:length(data_day) % loop of hours
        
        
        data219= data_day(j).data219struct; % create new variable to loop within the structure for data    
        
        % Checking which indices of 235 K are open
        
        data235= data_day(j).data235struct;
         inny=[];
        for kk=1:length(data235)
            if data235(kk).isopen==1
                inny(end+1)=kk;
              %else
               %   inny=[];
            end
        end
%         inny
        
        ins=[];
        % In this part an array saving values of all the indicies of  235K outside of the specidifc 219s at that hour
        
        for k=1:length(data219)
            if ~isempty(data219(k).inds235outside)
%             f=ismember(data219(k).inds235outside,inny)
%             if f==0
                ins(end+1)=data219(k).inds235outside(1); 
                % end
            end
            
        end
      %  ins
        %  Then the array will be used to identify the 235 indices that have only one 219K in a 235K
        d=unique(ins);
        Ncount=histc(ins,d);
        v=find(Ncount==1);
        xx=d(v);
        for l=1:length(xx);
            % if ismember(xx(l),inny)==0
            for m=1:length(data219)
                if data219(m).inds235outside==xx(l)
                    if ismember(data219(m).inds235outside,inny)==0
                        
                        if data219(m).area_km>=4000
                         
                            count=count+1;
                            
                            newsystems(count).year=data_day(j).year;
                            newsystems(count).month=data_day(j).month;
                            newsystems(count).day=data_day(j).day;
                            newsystems(count).hour=data_day(j).hour;
                            newsystems(count).minutes=data_day(j).minutes;
                            newsystems(count).seconds=data_day(j).seconds;
                            
                            %datetime strucutre
                            year =newsystems(count).year ;
                            month  =newsystems(count).month ;
                            day =newsystems(count).day;
                            hour =newsystems(count).hour;
                            minutes=newsystems(count).minutes;
                            seconds  =newsystems(count).seconds;
                            strin=[year, '-', month, '-', day, ' ', hour, ':', minutes, ':' seconds];
                            timenow=datetime(strin,'InputFormat','yyyy-MM-dd HH:mm:ss');
                            newsystems(count).datestr=timenow;
                              
                            newsystems(count).suma_areakm=data219(m).area_km;

                            % This will be to save all of the info of the 219 K area
                            newsystems(count).data219struct=data219(m);
                            %In this case since it is only one 219 dont make average sof variables E, Ave T and standard dev
                            %but make the array of the one value 
                            %newsystems(count).ave219E=[];
                            newsystems(count).avestd219=data219(m).std1real;
                            newsystems(count).avetemp219=data219(m).meantempreal;
                            newsystems(count).arearatios_ave=data219(m).area_km/data235(xx(l)).area_km; % not really ave cause its justone 219 K
                            newsystems(count).lat_dist=data235(xx(l)).centroid_ydist;
                            newsystems(count).lat=data235(xx(l)).centroid_y;
                            newsystems(count).lon=data235(xx(l)).centroid_x;
                            newsystems(count).lon_dist=data235(xx(l)).centroid_xdist;

		            % Lets reduce the points of x and y edges 
           			        [newx,newy]=reducem(data235(xx(l)).x,data235(xx(l)).y);
                            newsystems(count).x=newx; %data235(xx(l)).x;
                            newsystems(count).y=newy; %data235(xx(l)).y;
                            newsystems(count).area_km=data235(xx(l)).area_km;
                           
                            
                            %Trying fit_ellipse
                            
                            ellipse=fit_ellipse(data235(xx(l)).x,data235(xx(l)).y);
                            newsystems(count).eccen=real(sqrt(1- (((ellipse.b)^2)/((ellipse.a)^2))));

                            
                        end
                    end %para if nuevo
                end
            end
        end
        % end
        %     For 235K that have moree than 1 219K cluster inside 
        v2=find(Ncount~=1);
        xx2=d(v2);
        
        pu=0;
        index=0;
        data219sum=struct([]);
        suma_areakm=0;
        suma_E=0;
        suma_stand=0;
        suma_temp=0;
        ave_E=0;
        ave_stand=0;
        ave_temp=0;
        lengthy=[];
        for n=1:length(xx2);
            num=xx2(n);
            % if ismember(num,inny)==0
            for o=1:length(data219)
                % if ~isempty(data219(o).x)
                if data219(o).inds235outside==num
                    if ismember(data219(o).inds235outside,inny)==0
                        
                        lengthy(end+1)=o;
                        %if ~isempty(data219(o).x)
                        index=index+1;
                        % save index that are inds235outside to use the length of them aftertwards for calucualting means
                        

                        data219sum(index).isopen=data219(o).isopen;
                        data219sum(index).area1=data219(o).area1;
                        data219sum(index).x=data219(o).x;
                        data219sum(index).y=data219(o).y;

                        data219sum(index).meantempreal=data219(o).meantempreal;
                        data219sum(index).std1real=data219(o).std1real;
                        data219sum(index).area_km=data219(o).area_km;
                        data219sum(index).inds235outside=data219(o).inds235outside;
                        data219sum(index).Class=data219(o).Class;

                        data219sum(index).centroid_x=data219(o).centroid_x;
                        data219sum(index).centroid_y=data219(o).centroid_y;

                        data219sum(index).centroid_xdist=data219(o).centroid_xdist;
                        data219sum(index).centroid_ydist=data219(o).centroid_ydist;

                        
                        % Adding individual ratios  (singles area 219/ 235)
                        data219sum(index).arearatios_ind=data219(o).area_km/data235(xx2(n)).area_km;
                        
                        suma_areakm=suma_areakm+data219(o).area_km;
                        % Averaging Eccentricity of all 219K
                        %suma_E=suma_E+data219(o).eccentricity;
                        
                        % ave_E=suma_E/index;
                        %Averaging Temperatures of all 219K
                        suma_temp=suma_temp+ data219(o).meantempreal;
                        
                        ave_temp=suma_temp/index;  % Takes the mean temp oe a single 219 and at the end it takes the mean of means
                        % Averaging standard deviation
                        suma_stand=suma_stand+data219(o).std1real;
                        
                        ave_stand=suma_stand/index;

                    end
                end
                
                
                %    end
                
                newsum=suma_areakm;
            end
            if  newsum>=4000
                %  if ismember(num,inny)==0
                % only if 235 are NOT open
                %  if data235(xx2(n)).isopen==0
                count=count+1;
                
                newsystems(count).year=data_day(j).year;
                newsystems(count).month=data_day(j).month;
                newsystems(count).day=data_day(j).day;
                newsystems(count).hour=data_day(j).hour;
                newsystems(count).minutes=data_day(j).minutes;
                newsystems(count).seconds=data_day(j).seconds;
                
                %datetime strucutre
                year =newsystems(count).year ;
                month  =newsystems(count).month ;
                day =newsystems(count).day;
                hour =newsystems(count).hour;
                minutes=newsystems(count).minutes;
                seconds  =newsystems(count).seconds;
                strin=[year, '-', month, '-', day, ' ', hour, ':', minutes, ':' seconds];
                timenow=datetime(strin,'InputFormat','yyyy-MM-dd HH:mm:ss');
                newsystems(count).datestr=timenow;
                               
                newsystems(count).suma_areakm= newsum; %suma_areakm;

                % This will be to save all of the info of the 219 K area
                newsystems(count).data219struct=data219sum; % Here all of the information of all the 219K inside the 235K are saved
                              
                newsystems(count).avestd219=ave_stand;
                
                newsystems(count).avetemp219=ave_temp;
                             
                
                % Adding 235 info
                newsystems(count).lat_dist=data235(xx2(n)).centroid_ydist;
                newsystems(count).lat=data235(xx2(n)).centroid_y;
                newsystems(count).lon=data235(xx2(n)).centroid_x;
                newsystems(count).lon_dist=data235(xx2(n)).centroid_xdist;

                [newx,newy]=reducem(data235(xx2(n)).x,data235(xx2(n)).y);
		        newsystems(count).x=newx; %data235(xx2(n)).x;
                newsystems(count).y=newy; %data235(xx2(n)).y;
                newsystems(count).area_km=data235(xx2(n)).area_km;
                
                % Adding average ratios ( sum_area average of 219 over 235 area)
                newsystems(count).arearatios_ave=newsum/data235(xx2(n)).area_km;
                
                %ADD ELLIPSE INFO AND ECCENTRICITY OF 235
                
                ellipse=fit_ellipse(data235(xx2(n)).x,data235(xx2(n)).y);
                newsystems(count).eccen=real(sqrt(1- (((ellipse.b)^2)/((ellipse.a)^2)))); %ellipse.b/ellipse.a;

                
                
            end 
            data219sum=[];
            %  disp('n')
            %  disp(n)
            suma_areakm=0;
            suma_E=0;
            ave_E=0;
            suma_temp=0;
            ave_temp=0;
            suma_stand=0;
            ave_stand=0;
            index=0;
            % end
            
        end
    end
    % What day i, what CE j
    
    fprintf('\n\n i = %d, j = %d \n\n\n', i, j)
end
%end
disp('Time for putting all CE in 1 structure')
toc;

clear data_day
clear data219
clear data235


% Adding field value of "kids"
%% Make a "smaller strcuture only with dates and reduced edges of x and y to overlap to leave the "big structure" sitting

for p=1:length(newsystems)
    nnewsystems(p).globalkids=[]; %p;
    nnewsystems(p).globalparent=[]; 
    nnewsystems(p).index=p;
%%
	nnewsystems(p).y=[newsystems(p).y];
	nnewsystems(p).x=[newsystems(p).x];
	nnewsystems(p).year=[newsystems(p).year];
	nnewsystems(p).day=[newsystems(p).day];
	nnewsystems(p).month=[newsystems(p).month];
	nnewsystems(p).hour=[newsystems(p).hour];
	nnewsystems(p).minutes=[newsystems(p).minutes];
	nnewsystems(p).seconds=[newsystems(p).seconds];
	%nnewsystems(p).globalkids=[newsystems.globalkids];
	%nnewsystems(p).globalparent=[newsystems.globalparent]; 
	%nnewsystems(p).index=[newsystems.index];
 
end

%crash 
%% After 235K  contours are indentified they are compared by overlapping technique                                                                                                                                   
ind=1;
tic;
for r=1:length(nnewsystems)-1 % loop of current centroid not including the last one
    if ~isempty(nnewsystems(r).x)
%         disp('current CE')
%         r        
        %Scale ploygons to km and round to integer
        %now values% 6. Then overlapping is computed by taking the current contour and comparing it to all the contours of the next hours. If overlap between
%    the overlap region over the smaller contour is >= 50 they are merged and a "kid" is added to that contour where "kid" is ID 
%    of the other contour that matche with.
        y1=round(6371*(nnewsystems(r).y)*(pi/180)); % changing lat to km the edges 
        x1=round(6371*(nnewsystems(r).x)*(pi/180).*cosd(nnewsystems(r).y)); % changing lon to km and rounding
         % First project the edges of the current system in the future to 108 km west as a system using 15 m/s speed
        % (climatology) would travel if moving straight west no meridional component taken in to consideration
        %x1=x1-108; % prjection of edges on x 
        % Using a 10 m/s speed and thus 72 km west every 2 hours 
        % x1=x1-72;
        % Using a 5 m/s speed and thus 36 km west every 2 hours 
        %x1=x1-36;
        % Using a 12 m/s speed and thus 86.4km km west every 2 hours 
        %x1=x1-86;
        
        % Current Centroid
        %x1_centroid=round(6371*(newsystems(r).lon)*(pi/180).*cosd(newsystems(r).lat)); % First change to km 
        %Projection on x  of centroid location
       
        % x centroid projeciton in degerrs of longitude              
        %projected_lon= newsystems(r).lon + (108/6371) * (180/pi) /cosd(newsystems(r).lat);  % Note the 108 km dx in positive         
        
        % projection of centroid on x in km
        %x1_centroid=x1_centroid-64;
  
        %Date of current cloud element 
        yearnow =nnewsystems(r).year ;
        monthnow  =nnewsystems(r).month ;
        daynow =nnewsystems(r).day; 
        hournow =nnewsystems(r).hour;
        minutesnow=nnewsystems(r).minutes;
        secondsnow  =nnewsystems(r).seconds;
        strinnow=[yearnow, '-', monthnow, '-', daynow, ' ', hournow, ':', minutesnow, ':' secondsnow];
        timenow=datetime(strinnow,'InputFormat','yyyy-MM-dd HH:mm:ss');
        % Find systems that within 24 hours  are in range of current
        
        for s=r+1:length(nnewsystems);  % for loop that compares all next hours with current hour
            if ~isempty(nnewsystems(s).x)
                
                %datestring for next hour
                yearafter =nnewsystems(s).year ;
                monthafter  =nnewsystems(s).month ;
                dayafter  =nnewsystems(s).day;
                hourafter  =nnewsystems(s).hour;
                minutesafter =nnewsystems(s).minutes;
                secondsafter   =nnewsystems(s).seconds;
                strinafter=[yearafter, '-', monthafter, '-', dayafter, ' ', hourafter, ':', minutesafter, ':' secondsafter];
                timeafter=datetime(strinafter,'InputFormat','yyyy-MM-dd HH:mm:ss');
                
                diffy=timeafter-timenow;
                diffhour=hours(diffy);
                % At this point the 24 hour limit is done to match CEs that are 24 hour at most berween time to make the  code faster
               % if (diffhour> 24)
                if diffhour >1.0 % same hour check
                    if diffhour > 4.0 % too far in furre to care to check
                        break
                    end
                    % do overlap check with projected edges of current systems because our cases arwe 2 hours apart
                   % disp([r,s])
                    %disp ('hour diff')
                    %disp([diffhour])
               
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NOT GOING TO USE THIS PART%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    % Check if distance is the distance between theCLOud Elemtnes is the climatological distance
                    %                 real_dist = distance(newsystems(r).lat,newsystems(r).lon,newsystems(s).lat, newsystems(s).lon)*(pi/180)*6371;
                    %                 %%Using 13.5 m/s as climatological value since
                    %                 %%tipical average velocoty of MCS is 12-1 5 m/s
                    %                 % projections of now datat
                    %                 projectionx=newsystems(r).lon_dist+diffhour*120*newsystems(r).u; % projected position x direciton
                    %                 projectiony=newsystems(r).lat_dist+diffhour*120*newsystems(r).v; % projected position y direction
                    %                 projectionydeg=(projectiony/6371)*(180/pi);
                    %                 projectionxdeg=(projectionx/6371)*(180/pi)*(1/cosd(projectionydeg));
                    %                 climate_dist=distance(projectionydeg,projectionxdeg,newsystems(s).lat, newsystems(s).lon)*(pi/180)*6371;
                    %                 %sqrt((datanow(r).u)^2+(datanow(r).v)^2)*(diffhour*120)  %(13.5*diffhour*120)/1000; % changing hours to seconds and divide by 1000 to make it in km
                    %                 %                          if dist
                    %                 errorfactor=1.5;%1.006 ;%1.5 ;%1.01; %1.0; allwoing an error of 50 % from the actual dustance or a 45 degrrees angel range
                    %                 if climate_dist <=real_dist*errorfactor
                    %(climate_dist-errorfactor <=real_dist && )
                    % disp('true')
                    %(projectiony -0.05 <dataafter(s).lat && dataafter(s).lat< projectiony +0.05 && ...
                    %  projectionx -0.05 <dataafter(s).lon && dataafter(s).lon< projectionx +0.05)
                    %dataafter(s).lat, dataafter(s).lon)*(pi/180)*6371  %   ral_dist >= climate_dist -1/2;
                    % Change the u and v of the next position and thus s
                    %                     if diffhour~=0 % same time C E will give infintie                    velocities
                    %                         newsystems(r).u=(newsystems(s).lon_dist-newsystems(r).lon_dist)/(diffhour*120);                       %[ km/s ]
                    %                         newsystems(r).v=(newsystems(s).lat_dist-newsystems(r).lat_dist)/(diffhour*120);  %[ km/s ]
                    %                     end
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %disp('CDurrent future cE')
                    %s
                    x2=round(6371*(nnewsystems(s).x)*(pi/180).*cosd(nnewsystems(s).y)); % changing lon to km and rounding
                    y2=round(6371*(nnewsystems(s).y)*(pi/180)); % changing lat to km
                    
                    %%% Check how far away the projected centroid is from  the CE that is being comapred to 
                    %disp('Dist bet prj cent and  current future CE')
                    %disty=distance(newsystems(r).lat,projected_lon,newsystems(s).lat,newsystems(s).lon)*(pi/180)*6371
                    
                    % Making x2 and y2 same size as x1 and y1
                    [row,column]=size(x1);
                    [row2,column2]=size(x2);
                    rowss=[row,row2];
                    maxi=max(rowss);
                    mini=min(rowss);
                    xx2=nan(maxi,1);
                    yy2=nan(maxi,1);
                    xx1=nan(maxi,1);
                    yy1=nan(maxi,1);
                    if length(x1)==maxi
                        
                        xx2(1:mini)=x2(1:mini);
                        yy2(1:mini)=y2(1:mini);
                        xx1=x1;
                        yy1=y1;
                    else
                        
                        xx1(1:mini)=x1(1:mini);
                        yy1(1:mini)=y1(1:mini);
                        xx2=x2;
                        yy2=y2;
                    end
                    % Offset polygons to the fit in [1,rows,1,columnns]
                    % matrix
                    deltaX = min(min(xx1,xx2));
                    deltaY = min(min(yy1,yy2));
                    rows = max(max(yy1,yy2));
                    columns =max(max(xx1,xx2)); % added a round 
                    
                    if deltaX <= 0
                        xx1 = xx1-deltaX+1;
                        xx2 = xx2-deltaX+1;
                        columns = columns-deltaX+2;
                    end
                    
                    if deltaY <= 0
                        yy2 = yy2-deltaY+1;
                        yy1 = yy1-deltaY+1;
                        rows = rows-deltaY+2;
                    end
                    %Changing back to not havinf NaN
                    xx1=num2cell(xx1);
                    xx1(cellfun(@isnan,xx1))={[]} ;
                    yy1=num2cell(yy1);
                    yy1(cellfun(@isnan,yy1))={[]} ;
                    xx2=num2cell(xx2);
                    xx2(cellfun(@isnan,xx2))={[]} ;
                    yy2=num2cell(yy2);
                    yy2(cellfun(@isnan,yy2))={[]} ;
                    
                    xx1=cell2mat(xx1);
                    yy1=cell2mat(yy1);
                    xx2=cell2mat(xx2);
                    yy2=cell2mat(yy2);
                    
                    %  See what points are in the polygons and their
                    %  overlap
                    b1 = poly2mask(xx1, yy1, rows, columns);
                    
                    b2 = poly2mask(xx2, yy2, rows, columns);
%                     disp('time for poly')
%                     toc
                    overlap = b1 & b2;
%                      disp('time for & ')
%                      toc
%                     
                    % Map the results!
%                         figure
%                         contour(b1,1,'b')
%                         hold on
%                        % plot(x1,y1,'r*')
%                         hold on
%                        % plot(x2,y2,'*g')
%                         contour(b2,1,'r')
%                         contour(overlap,1,'m','Linewidth',2);
%                         hold off
%                     
                    
                    % compute paramter to see if overlap is big enough
                    size1 = sum(sum(b1));
                    size2 = sum(sum(b2));
                    sizeOverlap = sum(sum(overlap));  % overlap / smaller anvil
                    frac = sizeOverlap/min(size1,size2);
                    percenty=frac*100;
                     % disp([percenty])
                    if percenty>=55
                        nnewsystems(r).globalkids(end+1)=nnewsystems(s).index; %newsystems(s).kids;
                        nnewsystems(s).globalparent(end+1)=nnewsystems(r).index;
                      
                    end  
                end
            end
        end
        % log where we are
        disp('current systems that is being compared to')
        fprintf('\n\n r = %d ', r)
    end
end
disp('time all overlapping and projections is done to all systems')
toc;
disp('time rate all overlapping and projections is done to all systems')
runRate=toc/length(nnewsystems);
fprintf('\n\n runRate = %d ', runRate)
tic;
% Paste global kids and parents and index ack to newysstems
 for p=1:length(nnewsystems)
    newsystems(p).globalkids=nnewsystems(p).globalkids; %p;
    newsystems(p).globalparent=nnewsystems(p).globalparent;
    newsystems(p).index=nnewsystems(p).index;
end
 nnewsystems=[];
 
%% Organizing the systems  cd
[Systems]=organizer(newsystems); 
% crash
%% Here I call classification  to cllassify systems
[systems_classified]=classification_new_new(newsystems,Systems);  
%  
%% At thispoint I can erase newsystems from memory 
%% Here Make plots of systems and their families  NOTE: DO NOT COMMENT ONLY COMMENT PLOT OR IT CANT OBTAIN LOCAL KIDS AND PARENTS
[systems_classified]=familyplots(systems_classified);  

%% Here I call precipadd 
% we use the non stats one to make sure we assing precip to each defined edge
[systems_classified]=precipassign(systems_classified);   

crash 
[ systems_classified ] =freqplots( systems_classified ); % to get some time information and save on "Position" matfile 
%% Here I save all the matfiles 
[ systems_classified, Systemss, Positions ] = savingfiles(systems_classified,first_date,last_date,outputbasedir);
% Plottign rainfall
% xlimss =[-35 45];
% ylimss=[0, 30];
% figure
% plot([systems_classified(4).System(1).x], [systems_classified(4).System(1).y],'k-','LineWidth',2)
% hold on
% 
% delta = 0.035; % EUMETSAT ~ 3km
% targetLats =ylimss(1):delta:ylimss(2); targetLons = xlimss(1):delta:xlimss(2);
% [lonArray,latArray]=meshgrid(targetLons,targetLats);
% new_lon = lonArray.*systems_classified(4).Raindata(1).indices_in;
% new_lat = latArray.*systems_classified(4).Raindata(1).indices_in ;
% pcolor([new_lon],[new_lat],[systems_classified(4).Raindata(1).rain_rate_2])
% shading interp
% cmap=cbrewer('div','Spectral',64);
% colormap(flipud(cmap));
% set(gcf,'color','w')
% set(gca,'FontSize',15)
% hh=colorbar;
% title(hh,'[mm hr^{-1}]');
% title(' CCC identified  on 2006-09-11 00:00 UTC')
% xlabel('Longitude [Degrees east]')
% ylabel('Latitude [Degrees north]')
% xlim([24 34])
% ylim([9 18]) 
