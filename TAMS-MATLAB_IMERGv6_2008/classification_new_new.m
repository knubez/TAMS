function [ systems_classified] = classification_new_new(newsystems,Systems)
%Classification_new_new-name of function
%----------------
% 1. It will put each system in a new structure with all the information that is needed
% 2. Then, according to Evans and Shemo work, it classifyes the systems in two main groups : MCS or MCC dependeing on size and duration criteria
% 3. Disorganized systems ar enot followed in this tracking method (anything below 4,000km or with a livelhood of less than 3 hours)


% HIST
% Created: January  16th 2018 by Kelly M. Nunez Ocasio
% --------------------------------------------------------------------------------------------------------------------
tic;
systems_classified=struct();
systems_classified.System=struct();
for i=1:length(Systems)
    %Add family members to each row
    systems_classified(i).GlobalFamily=Systems(i).list;
    for j=1:length(Systems(i).list)
        %j
        systems_classified(i).System(j).data219struct=newsystems(Systems(i).list(j)).data219struct;
        systems_classified(i).System(j).year=newsystems(Systems(i).list(j)).year;
        systems_classified(i).System(j).month=newsystems(Systems(i).list(j)).month;
        systems_classified(i).System(j).day=newsystems(Systems(i).list(j)).day;
        systems_classified(i).System(j).hour=newsystems(Systems(i).list(j)).hour;
        systems_classified(i).System(j).minutes=newsystems(Systems(i).list(j)).minutes;
        systems_classified(i).System(j).seconds=newsystems(Systems(i).list(j)).seconds;
        systems_classified(i).System(j).datestr=newsystems(Systems(i).list(j)).datestr;     
        systems_classified(i).System(j).suma_areakm=newsystems(Systems(i).list(j)).suma_areakm;
        systems_classified(i).System(j).avestd219=newsystems(Systems(i).list(j)).avestd219;
        systems_classified(i).System(j).avetemp219=newsystems(Systems(i).list(j)).avetemp219;
        systems_classified(i).System(j).lat_dist=newsystems(Systems(i).list(j)).lat_dist;
        systems_classified(i).System(j).lat=newsystems(Systems(i).list(j)).lat;
        systems_classified(i).System(j).lon=newsystems(Systems(i).list(j)).lon;
        systems_classified(i).System(j).lon_dist=newsystems(Systems(i).list(j)).lon_dist;
        systems_classified(i).System(j).x=newsystems(Systems(i).list(j)).x;
        systems_classified(i).System(j).y=newsystems(Systems(i).list(j)).y;
        
        systems_classified(i).System(j).area_km=newsystems(Systems(i).list(j)).area_km;
        systems_classified(i).System(j).arearatios_ave=newsystems(Systems(i).list(j)).arearatios_ave;
        % Ellipse info
        systems_classified(i).System(j).eccen=newsystems(Systems(i).list(j)).eccen;

        
        %Info on kids and parents
        systems_classified(i).System(j).globalkids=newsystems(Systems(i).list(j)).globalkids;
        systems_classified(i).System(j).globalparent=newsystems(Systems(i).list(j)).globalparent;
        
        
    end
end
% CLASSIFICATION
for m=1:length(systems_classified)
    classy=systems_classified(m).System;
    %length(classy)
    maxbcrcount=0;
    maxsccount=0;
    
    bcrcount=[];
    sccount=[];
    % Variables bcrcount_dsl and sccount_dsl
    %are for cases that are ever growing that can potentially be a TC . The algorithm doesnt no detect its end so it
    % will classfy it as a DSL so I have to check two times the DSL to make sure is not A TC or long-lived systems
    
    bcrcount_dsl=[];
    sccount_dsl=[];
    first_day=0;
    diff_hour=0;
    diff_hour_2=0;
    for n=1:length(classy)
        if n==1
            if (classy(n).area_km >=5.0000e+04)  &&  (classy(n).eccen<=0.7) && (classy(n).suma_areakm >=2.5000e+04)
            %if (classy(n).area_km >=5.0000e+04)  &&  (classy(n).eccen2>=0.7) && (classy(n).suma_areakm >=2.5000e+04)  
                bcrcount(end+1)=n;
                bcrcount_dsl(end+1)=n;
                
                % save 1 inbcrcount field
                systems_classified(m).System(n).countbigcold=1;
                
            else
                bcrcount=[];
            end
            
        else
            if (classy(n).area_km >=5.0000e+04)  &&  (classy(n).eccen<=0.7) && (classy(n).suma_areakm >=2.5000e+04)
            %if (classy(n).area_km >=5.0000e+04)  &&  (classy(n).eccen2>=0.7) && (classy(n).suma_areakm >=2.5000e+04)
                bcrcount(end+1)=n;
                % For those systems that are long-lived but may be still categorize as DSL check the first 1 of the systems
                % to see if it was CCC or MCC
                bcrcount_dsl(end+1)=n;
                
                
                % save 1 inbcrcount field
                systems_classified(m).System(n).countbigcold=1;
                
            else
                
                if str2double(classy(n).hour)~=str2double(classy(n-1).hour)
                    if ~isempty(bcrcount)
                        systems_classified(m).System(n).countbigcold=0;
                        lengthy=length(bcrcount);
                        %                     % Final time conversion from index to time %
                        Year_f=classy((bcrcount(lengthy))).year; %
                        Month_f=classy(bcrcount(lengthy)).month; %
                        Day_f=classy(bcrcount(lengthy)).day; %
                        Hour_f=classy(bcrcount(lengthy)).hour; %
                        Minutes_f=classy(bcrcount(lengthy)).minutes; %
                        Seconds_f=classy(bcrcount(lengthy)).seconds; %                     %
                        % making a string for datetime %
                        string_f=[Year_f,'-', Month_f,'-',Day_f,' ', Hour_f,':',Minutes_f,':',Seconds_f ]; %
                        
                        stringy_f=datetime(string_f,'InputFormat','yyyy-MM-dd HH:mm:ss'); % %
                        % Intial time converison from index to time %
                        Year_i=classy(bcrcount(1)).year; %
                        Month_i=classy(bcrcount(1)).month; %
                        Day_i=classy(bcrcount(1)).day; %
                        Hour_i=classy(bcrcount(1)).hour; %
                        Minutes_i=classy(bcrcount(1)).minutes; %
                        Seconds_i=classy(bcrcount(1)).seconds; %
                        % makinga string for datetime %
                        string_i=[Year_i,'-', Month_i,'-',Day_i,' ', Hour_i,':',Minutes_i,':' Seconds_i ]; %
                        stringy_i=datetime(string_i,'InputFormat','yyyy-MM-dd HH:mm:ss'); % %
                        % Substrating the last time from the first to know for how long the
                        %system lived %
                        Duration1=stringy_f-stringy_i;
                        diff_hour=hours(Duration1); % % % %                     %
                        %diff_hour=classy(bcrcount(lengthy)).hour-classy((bcrcount(1))).hour;
                        maxbcrcount=max(maxbcrcount,diff_hour); %
                        bcrcount=[];
                    end
                else
                    %maxbcrcount=max(maxbcrcount,diff_hour);
                    if ~isempty(bcrcount)
                        lengthy=length(bcrcount);
                        %                     % Final time conversion from index to time %
                        Year_f=classy((bcrcount(lengthy))).year; %
                        Month_f=classy(bcrcount(lengthy)).month; %
                        Day_f=classy(bcrcount(lengthy)).day; %
                        Hour_f=classy(bcrcount(lengthy)).hour; %
                        Minutes_f=classy(bcrcount(lengthy)).minutes; %
                        Seconds_f=classy(bcrcount(lengthy)).seconds; %                     %
                        % making a string for datetime %
                        string_f=[Year_f,'-', Month_f,'-',Day_f,' ', Hour_f,':',Minutes_f,':',Seconds_f ]; %
                        
                        stringy_f=datetime(string_f,'InputFormat','yyyy-MM-dd HH:mm:ss'); % %
                        % Intial time converison from index to time %
                        Year_i=classy(bcrcount(1)).year; %
                        Month_i=classy(bcrcount(1)).month; %
                        Day_i=classy(bcrcount(1)).day; %
                        Hour_i=classy(bcrcount(1)).hour; %
                        Minutes_i=classy(bcrcount(1)).minutes; %
                        Seconds_i=classy(bcrcount(1)).seconds; %
                        % makinga string for datetime %
                        string_i=[Year_i,'-', Month_i,'-',Day_i,' ', Hour_i,':',Minutes_i,':' Seconds_i ]; %
                        stringy_i=datetime(string_i,'InputFormat','yyyy-MM-dd HH:mm:ss'); % %
                        % Substrating the last time from the first to know for how long the
                        %system lived %
                        Duration1=stringy_f-stringy_i;
                        diff_hour=hours(Duration1); % % % %                     %
                        %diff_hour=classy(bcrcount(lengthy)).hour-classy((bcrcount(1))).hour;
                        maxbcrcount=max(maxbcrcount,diff_hour); %
                        % bcrcount=[];
                    end
                end
            end
        end
        
        %only ccc
        if n==1
            if (classy(n).suma_areakm >=2.5000e+04) %(classy(n).suma_areakm >=2.5000e+04)
                sccount(end+1)=n;
                sccount_dsl(end+1)=n;
                % save 1 inbcrcount field
                systems_classified(m).System(n).countsmallcold=1;
            else
                sccount=[];
            end
        else
            if (classy(n).suma_areakm >=2.5000e+04)%classy(n).suma_areakm >=2.5000e+04
                sccount(end+1)=n;
                % For those systems that are long-lived but may be still categorize as DSL because of smaller cloud elemetns at the same time not meeting criteria check the first                   of the systems
                % to see if it was CCC or MCC
                sccount_dsl(end+1)=n;
                
                
                %save
                systems_classified(m).System(n).countsmallcold=1;
                
            else
                if str2double(classy(n).hour)~=str2double(classy(n-1).hour)
                    if ~isempty(sccount)
                        sccount;
                        systems_classified(m).System(n).countsmallcold=0;
                        lengthy_2=length(sccount);
                        % if ~isempty(sccount)
                        % Final time conversion from index to time
                        Year_f=classy(sccount(lengthy_2)).year;
                        Month_f=classy(sccount(lengthy_2)).month;
                        Day_f=classy(sccount(lengthy_2)).day;
                        Hour_f=classy(sccount(lengthy_2)).hour;
                        Minutes_f=classy(sccount(lengthy_2)).minutes;
                        Seconds_f=classy(sccount(lengthy_2)).seconds;
                        % making a string for datetime
                        string_f=[Year_f,'-', Month_f,'-',Day_f,' ', Hour_f,':',Minutes_f,':' Seconds_f ];
                        stringy_f=datetime(string_f,'InputFormat','yyyy-MM-dd HH:mm:ss');
                        
                        % Intial time converison from index to time
                        Year_i=classy(sccount(1)).year;
                        Month_i=classy(sccount(1)).month;
                        Day_i=classy(sccount(1)).day;
                        Hour_i=classy(sccount(1)).hour;
                        Minutes_i=classy(sccount(1)).minutes;
                        Seconds_i=classy(sccount(1)).seconds;
                        % making a string for datetime
                        string_i=[Year_i,'-', Month_i,'-',Day_i,' ', Hour_i,':',Minutes_i,':' Seconds_i ];
                        stringy_i=datetime(string_i,'InputFormat','yyyy-MM-dd HH:mm:ss');
                        
                        % Substrating the last time from the first to know for how long the system lived
                        Duration1=stringy_f-stringy_i;
                        diff_hour_2=hours(Duration1);
                        
                        maxsccount=max(maxsccount,diff_hour_2);
                        sccount=[];
                        % bcrcount=[];
                    end
                else
                    %maxsccount=max(maxsccount,diff_hour_2);
                    systems_classified(m).System(n).countsmallcold=0;
                    if ~isempty(sccount)
                        lengthy_2=length(sccount);
                        % if ~isempty(sccount)
                        % Final time conversion from index to time
                        Year_f=classy(sccount(lengthy_2)).year;
                        Month_f=classy(sccount(lengthy_2)).month;
                        Day_f=classy(sccount(lengthy_2)).day;
                        Hour_f=classy(sccount(lengthy_2)).hour;
                        Minutes_f=classy(sccount(lengthy_2)).minutes;
                        Seconds_f=classy(sccount(lengthy_2)).seconds;
                        % making a string for datetime
                        string_f=[Year_f,'-', Month_f,'-',Day_f,' ', Hour_f,':',Minutes_f,':' Seconds_f ];
                        stringy_f=datetime(string_f,'InputFormat','yyyy-MM-dd HH:mm:ss');
                        
                        % Intial time converison from index to time
                        Year_i=classy(sccount(1)).year;
                        Month_i=classy(sccount(1)).month;
                        Day_i=classy(sccount(1)).day;
                        Hour_i=classy(sccount(1)).hour;
                        Minutes_i=classy(sccount(1)).minutes;
                        Seconds_i=classy(sccount(1)).seconds;
                        % making a string for datetime
                        string_i=[Year_i,'-', Month_i,'-',Day_i,' ', Hour_i,':',Minutes_i,':' Seconds_i ];
                        stringy_i=datetime(string_i,'InputFormat','yyyy-MM-dd HH:mm:ss');
                        
                        % Substrating the last time from the first to know for how long the system lived
                        Duration1=stringy_f-stringy_i;
                        diff_hour_2=hours(Duration1);
                        
                        maxsccount=max(maxsccount,diff_hour_2);
                    end
                    
                end
                
            end
            
        end
    end
    
    systems_classified(m).Class='DSL';
    if maxsccount>= 6
        systems_classified(m).Class='CCC';
    end
    if maxbcrcount>= 6 % && macsccount>=6
        systems_classified(m).Class='MCC';
    end
    %%%% For those systems that are categotizes as DSL( because of systems having more than 1 "kid" for a specific hour ) but may be long-lived or TC and thus, more than a day %%%%
    %%%  Check if DSL is an MCC  %%%
    
    bcrcount_dsl;
    sccount_dsl;
    if  strcmp(systems_classified(m).Class,'DSL')==1
        if ~isempty(bcrcount_dsl)
            dum=0;
            hoursy_MCC=nan(1,length(bcrcount_dsl));
            for k=1:length(bcrcount_dsl)
                if str2double(classy(bcrcount_dsl(1)).day)==str2double(classy(bcrcount_dsl(k)).day)
                    %dips('kelly')
                    dum=dum+1;
                    hoursy_MCC(dum)=str2double(classy(bcrcount_dsl(k)).hour);
                end
            end
            true_hoursy_MCC=[hoursy_MCC(1):2:hoursy_MCC(length(hoursy_MCC))];
            dd=ismember(true_hoursy_MCC,hoursy_MCC);
            ff=find(dd==0);
            count=0;
            g=[];
            for q=1:length(hoursy_MCC)
                
                if ismember(q,ff)==0
                    g(end+1)=hoursy_MCC(q);
                else
                    g(end+1)=hoursy_MCC(q);
                    % g=[];
                    %if ~isempty(g)
                    count=max(count,g(length(g))-g(1));
                    g=[];
                    %end
                    
                end
                if length(g)>1
                    count=max(count,g(length(g))-g(1));
                end
                
            end
            
            count;
            if count>=6
                systems_classified(m).Class='MCC';
            end
        end
        %%%  Check if DSL is an cCC  %%%
        if ~isempty(sccount_dsl)
            dumy=0;
            hoursy_CCC=nan(1,length(sccount_dsl));
            for l=1:length(sccount_dsl)
                
                if str2double(classy(sccount_dsl(1)).day)==str2double(classy(sccount_dsl(l)).day)
                    dumy=dumy+1;
                    hoursy_CCC(dumy)=str2double(classy(sccount_dsl(l)).hour);
                end
            end
            true_hoursy_CCC=[hoursy_CCC(1):2:hoursy_CCC(length(hoursy_CCC))];
            dd=ismember(true_hoursy_CCC,hoursy_CCC);
            ff=find(dd==0);
            cuenta=0;
            a=[];
            for r=1:length(hoursy_CCC)
                
                if ismember(r,ff)==0
                    a(end+1)=hoursy_CCC(r);
                else
                    a(end+1)=hoursy_CCC(r);
                    % a=[];
                    %if ~isempty(g)
                    cuenta=max(cuenta,a(length(a))-a(1));
                    a=[];
                    %end
                    
                end
                if length(a)>1
                    cuenta=max(cuenta,a(length(a))-a(1));
                end
            end
            
            cuenta;
            if cuenta>=6
                systems_classified(m).Class='CCC';
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
        end
    end
    % In the case the DSL is still DSL check if its longer than 6 hours if
    % so it becomes a DLL , a long lived disorganized system, that doesnt meet the area criterion for MCC or CCC for 6 hours or more but does
    % last 6 hours or more with cold cloud tops (219 areas) bigger than 4,000km which is the minimum criteria for a cloud element
    % to become available for clasificaiton under organized
    if  strcmp(systems_classified(m).Class,'DSL')==1
        % Final time conversion from index to time
        Year_f=systems_classified(m).System(length(systems_classified(m).System)).year;
        Month_f=systems_classified(m).System(length(systems_classified(m).System)).month;
        Day_f=systems_classified(m).System(length(systems_classified(m).System)).day;
        Hour_f=systems_classified(m).System(length(systems_classified(m).System)).hour;
        Minutes_f=systems_classified(m).System(length(systems_classified(m).System)).minutes;
        Seconds_f=systems_classified(m).System(length(systems_classified(m).System)).seconds;
        % making a string for datetime
        string_f=[Year_f,'-', Month_f,'-',Day_f,' ', Hour_f,':',Minutes_f,':' Seconds_f ];
        stringy_f=datetime(string_f,'InputFormat','yyyy-MM-dd HH:mm:ss');
        
        % Intial time converison from index to time
        Year_i=systems_classified(m).System(1).year;
        Month_i=systems_classified(m).System(1).month;
        Day_i=systems_classified(m).System(1).day;
        Hour_i=systems_classified(m).System(1).hour;
        Minutes_i=systems_classified(m).System(1).minutes;
        Seconds_i=systems_classified(m).System(1).seconds;
        % making a string for datetime
        string_i=[Year_i,'-', Month_i,'-',Day_i,' ', Hour_i,':',Minutes_i,':' Seconds_i ];
        stringy_i=datetime(string_i,'InputFormat','yyyy-MM-dd HH:mm:ss');
        
        % Substrating the last time from the first to know for how long the system lived
        Duration1=stringy_f-stringy_i;
        diff_hour_2=hours(Duration1);
        
        if diff_hour_2>=6
            systems_classified(m).Class='DLL';
        end
        
    end
        
   
end

disp('Classification is done')
toc;
    % Because a system may still be growing the algortihm doesnt detetect its end so it will return a DSL
    % (this can potentilally be a hurricane , howEver we double check sccount and bcrcount to see if the last time and first time are
    % greater than 6 hours to classify it as either MCC or CCC
%     if strcmp(systems_classified(m).Class,'DSL')==1
%         if ~isempty(sccount)
%             lengthy_2=length(sccount);
%             % if ~isempty(sccount)
%             % Final time conversion from index to time
%             Year_f=classy(sccount(lengthy_2)).year;
%             Month_f=classy(sccount(lengthy_2)).month;
%             Day_f=classy(sccount(lengthy_2)).day;
%             Hour_f=classy(sccount(lengthy_2)).hour;
%             Minutes_f=classy(sccount(lengthy_2)).minutes;
%             Seconds_f=classy(sccount(lengthy_2)).seconds;
%             % making a string for datetime
%             string_f=[Year_f,'-', Month_f,'-',Day_f,' ', Hour_f,':',Minutes_f,':' Seconds_f ];
%             stringy_f=datetime(string_f,'InputFormat','yyyy-MM-dd HH:mm:ss');
%             
%             % Intial time converison from index to time
%             Year_i=classy(sccount(1)).year;
%             Month_i=classy(sccount(1)).month;
%             Day_i=classy(sccount(1)).day;
%             Hour_i=classy(sccount(1)).hour;
%             Minutes_i=classy(sccount(1)).minutes;
%             Seconds_i=classy(sccount(1)).seconds;
%             % making a string for datetime
%             string_i=[Year_i,'-', Month_i,'-',Day_i,' ', Hour_i,':',Minutes_i,':' Seconds_i ];
%             stringy_i=datetime(string_i,'InputFormat','yyyy-MM-dd HH:mm:ss');
%             
%             % Substrating the last time from the first to know for how long the system lived
%             Duration1=stringy_f-stringy_i;
%             diff_hour_2=hours(Duration1);
%         end
%         if diff_hour_2>=6
%             systems_classified(m).Class='CCC';
%         end
%         if ~isempty(bcrcount)
%             lengthy=length(bcrcount);
%             %                     % Final time conversion from index to time %
%             Year_f=classy((bcrcount(lengthy))).year; %
%             Month_f=classy(bcrcount(lengthy)).month; %
%             Day_f=classy(bcrcount(lengthy)).day; %
%             Hour_f=classy(bcrcount(lengthy)).hour; %
%             Minutes_f=classy(bcrcount(lengthy)).minutes; %
%             Seconds_f=classy(bcrcount(lengthy)).seconds; %                     %
%             % making a string for datetime %
%             string_f=[Year_f,'-', Month_f,'-',Day_f,' ', Hour_f,':',Minutes_f,':',Seconds_f ]; %
%             
%             stringy_f=datetime(string_f,'InputFormat','yyyy-MM-dd HH:mm:ss'); % %
%             % Intial time converison from index to time %
%             Year_i=classy(bcrcount(1)).year; %
%             Month_i=classy(bcrcount(1)).month; %
%             Day_i=classy(bcrcount(1)).day; %
%             Hour_i=classy(bcrcount(1)).hour; %
%             Minutes_i=classy(bcrcount(1)).minutes; %
%             Seconds_i=classy(bcrcount(1)).seconds; % 
%             % makinga string for datetime %
%             string_i=[Year_i,'-', Month_i,'-',Day_i,' ', Hour_i,':',Minutes_i,':' Seconds_i ]; %
%             stringy_i=datetime(string_i,'InputFormat','yyyy-MM-dd HH:mm:ss'); % %
%             % Substrating the last time from the first to know for how long the
%             %system lived %
%             Duration1=stringy_f-stringy_i;
%             diff_hour=hours(Duration1); % % % %                     %
%             %diff_hour=classy(bcrcount(lengthy)).hour-classy((bcrcount(1))).hour;
%             maxbcrcount=max(maxbcrcount,diff_hour); %
%             % bcrcount=[];
%         end
%         if diff_hour>= 6
%             systems_classified(m).Class='MCC';
%         end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% count=[];
% 
% for m=1:length(systems_classified)
%     classy=systems_classified(m).System;
%     
%     for n=1:length(classy)
%         if (classy(n).areakm) >=4.0000e+04  % First criteria of 219K % 1.5
%             count(end+1)=n;
%             %  lengthclassy=(length(classy));
%             
%             if ~isempty(count)
%                 % count
%                 hours_incount={classy(count).hour};
%                 hours_incount=unique(hours_incount); % so we dont repeat hours
%                 hours_incount=str2double(hours_incount);
%                 first_true={classy(count(1)).hour};
%                 first_true=str2double(first_true);
%                 last_true={classy(count(length(count))).hour};
%                 last_true=str2double(last_true);
%                 true_hours=[first_true:2:last_true];
%                 if length(hours_incount)==length(true_hours)
%                     count=count;
%                 else
%                     count=[];
%                 end
%             end
%             
%             
%             lengthy=length(count);
%             if ~isempty(count) % if 15000 km for the 219 areas is not satisfied in any of the times of that system it is DSL
%                 % Final time conversion from index to time
%                 Year_f=classy(count(lengthy)).year;
%                 Month_f=classy(count(lengthy)).month;
%                 Day_f=classy(count(lengthy)).day;
%                 Hour_f=classy(count(lengthy)).hour;
%                 Minutes_f=classy(count(lengthy)).minutes;
%                 Seconds_f=classy(count(lengthy)).seconds;
%                 % making a string for datetime
%                 string_f=[Year_f,'-', Month_f,'-',Day_f,' ', Hour_f,':',Minutes_f,':' Seconds_f ];
%                 stringy_f=datetime(string_f,'InputFormat','yyyy-MM-dd HH:mm:ss');
%                 
%                 % Intial time converison from index to time
%                 Year_i=classy(count(1)).year;
%                 Month_i=classy(count(1)).month;
%                 Day_i=classy(count(1)).day;
%                 Hour_i=classy(count(1)).hour;
%                 Minutes_i=classy(count(1)).minutes;
%                 Seconds_i=classy(count(1)).seconds;
%                 % making a string for datetime
%                 string_i=[Year_i,'-', Month_i,'-',Day_i,' ', Hour_i,':',Minutes_i,':' Seconds_i ];
%                 stringy_i=datetime(string_i,'InputFormat','yyyy-MM-dd HH:mm:ss');
%                 
%                 % Substrating the last time from the first to know for how long the system lived
%                 Duration1=stringy_f-stringy_i;
%                 Hours=hours(Duration1);
%                 
%                 
%                 
%                 if Hours>=6  %(first hour criteria fro areas 235K & 219 K)
%                     count_2=[];
%                     %dum=0;
%                     %values=[];
%                     for o=1:length(classy)
%                         disp('classy area')
%                         classy(o).area_km
%                         disp('classy eccen')
%                         classy(o).eccen
%                         if ([classy(o).area_km]) >= 4.0000e+04  &&  ([classy(o).eccen]) <= 0.7  % >=0.7 if 1 means circular % Second criteria of 235K to be MCC or CC     %2.5000e+04
%                             
%                             count_2(end+1)=o;
%                             
%                         end
%                     end
%                     if ~isempty(count_2)
%                         hours_incount_2={classy(count_2).hour};
%                         hours_incount_2=unique(hours_incount_2); % so we dont repeat hours
%                         hours_incount_2=str2double(hours_incount_2);
%                         first_true_2={classy(count_2(1)).hour};
%                         first_true_2=str2double(first_true_2);
%                         last_true_2={classy(count_2(length(count_2))).hour};
%                         last_true_2=str2double(last_true_2);
%                         true_hours_2=[first_true_2:2:last_true_2];
%                         if length(hours_incount_2)==length(true_hours_2)
%                             count_2=count_2;
%                         else
%                             count_2=[];
%                         end
%                     end
%                     
%                     if ~isempty(count_2)
%                         
%                         lengthy_2=length(count_2);
%                         % Final time conversion from index to time
%                         Year_f=classy(count_2(lengthy_2)).year;
%                         Month_f=classy(count_2(lengthy_2)).month;
%                         Day_f=classy(count_2(lengthy_2)).day;
%                         Hour_f=classy(count_2(lengthy_2)).hour;
%                         Minutes_f=classy(count_2(lengthy_2)).minutes;
%                         Seconds_f=classy(count_2(lengthy_2)).seconds;
%                         % making a string for datetime
%                         string_f=[Year_f,'-', Month_f,'-',Day_f,' ', Hour_f,':',Minutes_f,':' Seconds_f ];
%                         stringy_f=datetime(string_f,'InputFormat','yyyy-MM-dd HH:mm:ss');
%                         
%                         % Intial time converison from index to time
%                         Year_i=classy(count_2(1)).year;
%                         Month_i=classy(count_2(1)).month;
%                         Day_i=classy(count_2(1)).day;
%                         Hour_i=classy(count_2(1)).hour;
%                         Minutes_i=classy(count_2(1)).minutes;
%                         Seconds_i=classy(count_2(1)).seconds;
%                         % making a string for datetime
%                         string_i=[Year_i,'-', Month_i,'-',Day_i,' ', Hour_i,':',Minutes_i,':' Seconds_i ];
%                         stringy_i=datetime(string_i,'InputFormat','yyyy-MM-dd HH:mm:ss');
%                         
%                         % Substrating the last time from the first to know for how long the system lived
%                         Duration2=stringy_f-stringy_i;
%                         Hours_2=hours(Duration2);
%                         %  end
%                         
%                         % Chech wether or not it is MCC(meets 235K and 219K criterias of area during 6 or more hours)
%                         %or CCC (meets 219K criterion of area during 6 or more hours
%                         if Hours_2>=6
%                             systems_classified(m).Class='MCC';
%                         else
%                             systems_classified(m).Class='CCC';
%                             
%                         end
%                     else
%                         systems_classified(m).Class='CCC';
%                         
%                     end
%                     
%                 else % if it doesnt meet the areas and hour (first hour criteria) criteria then is a disorganized system
%                     
%                     systems_classified(m).Class='DSL';
%                 end
%             else % if it doesnt meet the areas and hour (first hour criteria) criteria then is a disorganized system
%                 
%                 systems_classified(m).Class='DSL';
%             end
%             
%         end
%         
%         


    

