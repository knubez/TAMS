function [ systems_classified ] =freqplots( systems_classified )
%freqplots-name of function
%----------------
% 1.First it will obtain the extremes of the paramaters for each system (genesis and termination times, 
%   min eccentricity, min temp (219), max stand deviation, max area).
% 2. Then it will saved them outside of each systems ffor under
%    systems_classified
% 3. Then it will proceed to plot them for each category 
% 4. It als calculates Duration of each system, speed of the system as whole ( final and intial position over duaration) 
%    max rainrate and rain acc for extremes ) . 

% HIST
% Created January  28th 2018 by Kelly M. Nunez Ocasio
% --------------------------------------------------------------------------------------------------------------------
% locations of files 
addpath('/gpfs/group/jle7/default/kmn18/graduateresearch/matlabcodes/altmany-export_fig-2763b78'); % add export_fig
% str={'CCC','MCC'};
cuenta=0;
contar=0;
tic;
for j=1:length(systems_classified)
    j;
    % 1) Find the maximum area time for each system
    variable_area=[systems_classified(j).System.area_km];
    index=find(variable_area==max(variable_area));
    % Save that hour in the big structure
    systems_classified(j).maxareatime=str2double(systems_classified(j).System(index).hour);
    
    % 2) Find maximum standard deviation (219) time- measures the spatial variablity of the cloud-top temp: high standard deviation indicates embedded towers
    % are present for that time ,increasing standard deviation implies vertical growth . Low standard deviation indictates  stratus or
    % cirrus clouds are likely , stable temp.
    variable_stand=[systems_classified(j).System.avestd219];
    index=find(variable_stand==max(variable_stand));
    % Save that hour in the big structure
    systems_classified(j).maxstd219time=str2double(systems_classified(j).System(index).hour);
    
    % 3) Find minimum temp time- lower cloud- top temps correpsond to higher  cloud-top heights
    variable_tempmin219=[systems_classified(j).System.avetemp219];
    index=find(variable_tempmin219==min(variable_tempmin219));
    % Save that hour in the big structure
    systems_classified(j).tempmin219time=str2double(systems_classified(j).System(index).hour);
    
    % 4)  Find minimum eccentricity time -measure of the shape of the system , because eccentricity is measured based on ellipses fitted to the
    % cloud elelemtn,anything <0.7 ismore organized/ convective and thus, more circular. The minimum of eccenwill tell when it was more ciucular 
    % ALways use the first time the eccen is small as it is usually paralle
    % to the other parameters 
    variable_eccen=[systems_classified(j).System.eccen]; 
    index=find(variable_eccen==min(variable_eccen));
    if length(index)>1
        %NOTE : Because you may have times where a small blob is very circular but not the main system index is probably more than one time so you
         %take the time when the are aif the system was greatest
        r=find([systems_classified(j).System.area_km]==max([systems_classified(j).System(index).area_km]));
        systems_classified(j).eccenmintime=str2double(systems_classified(j).System(r).hour);
    else
        systems_classified(j).eccenmintime=str2double(systems_classified(j).System(index).hour);
    end

    % 5) Save genesis time of the systems 
    systems_classified(j).genesistime=str2double(systems_classified(j).System(1).hour);
    % 6) Save termination time of systems
    systems_classified(j).terminationtime=str2double(systems_classified(j).System(length(systems_classified(j).System)).hour);
    
    % Becasue DSL dont have rainfall asignment 
    if ~isempty([systems_classified(j).Raindata])
        %display date of of raindata and system classified 
        j
        fecharain = [systems_classified(j).Raindata.filebasename]
        fechasystem =[systems_classified(j).System.datestr]   
        
        % 7) Save time of maximum area sum rain accumulation
        variable_sumrainacc=[systems_classified(j).Raindata.area_sum_rain_acc]
        index=find(variable_sumrainacc==max(variable_sumrainacc))
        
        if isempty(index) | isnan(index)

          systems_classified(j).max_area_sum_rain_acc_time=[];
       else 

       	 % Save that hour in the big structure
         systems_classified(j).max_area_sum_rain_acc_time=str2double(systems_classified(j).System(index).hour);
        end

        % 8) Save time of maximum area average rain rate
        variable_averainrate=[systems_classified(j).Raindata.area_ave_rain_rate];
        index=find(variable_averainrate==max(variable_averainrate));

        if isempty(index) | isnan(index)
            systems_classified(j).max_area_ave_rain_rate_time=[];
        else

       	     % Save that hour in the big structure
            systems_classified(j).max_area_ave_rain_rate_time=str2double(systems_classified(j).System(index).hour);
        end
    end
    
        
    
    
    
    
    
    
    
%    
%     %Save time when the most percentage of system was precipitating 
%     if ~isempty(systems_classified(j).Raindata)
%         variable_rain={systems_classified(j).Raindata.rainfrac}; % Here we do cell because of rows are empty inisde Raindata
%         maxy_frac=max([systems_classified(j).Raindata.rainfrac])
%         disp('month')
%         systems_classified(j).Raindata.Month
%         disp('day') 
%         systems_classified(j).Raindata.Day
%         
%         if maxy_frac==0 %when trmm doesn thave rian 
%             systems_classified(j).rainfracmax=[];
%         else
%          
%             ismax = cellfun(@(x)isequal(x,maxy_frac),variable_rain)
%             ind=find(ismax)
%             systems_classified(j).rainfracmax=str2double(systems_classified(j).Raindata(ind(1)).Hour);  % Note for frac we use thefirst max frac ,
%         end
%         %this happens when there are hte mae vlaue in two places
%     end 
%     
%     % Sve time when maximu average raintrate of precipitaitng area
%      if ~isempty(systems_classified(j).Raindata)
%          if maxy_frac~=0
%              %When  all  values are nANs and thus there is no rain detected
%              variable_rainrate={systems_classified(j).Raindata.ave_rainrate_area}; % Here we do cell because of rows are empty inisde Raindata                       
%              maxy=max([systems_classified(j).Raindata.ave_rainrate_area]);            
%              ismax = cellfun(@(x)isequal(x,maxy),variable_rainrate);
%              ind=find(ismax);     
%              systems_classified(j).ave_rainrate_area=str2double(systems_classified(j).Raindata(ind).Hour);
%          else
%              maxy=0;%when there is no rain within system or to less of rain fro TRMM to capture
%              systems_classified(j).ave_rainrate_area= [] ; %str2double(systems_classified(j).Raindata(ind).Hour);
%          end
%      end
%       % Sve time when maximum average rainaccumulaiton of precipitaitng area
%      if ~isempty(systems_classified(j).Raindata)
%          if maxy_frac~=0
%              variable_rainacc={systems_classified(j).Raindata.ave_rainacc_area}; % Here we do cell because of rows are empty inisde Raindata
%              maxy=max([systems_classified(j).Raindata.ave_rainacc_area]);
%              ismax = cellfun(@(x)isequal(x,maxy),variable_rainacc);
%              ind=find(ismax);
%              systems_classified(j).ave_rainacc_area=str2double(systems_classified(j).Raindata(ind).Hour);
%          else
%              maxy=0; %when there is no rain within system or to less of rain fro TRMM to capture
%              systems_classified(j).ave_rainacc_area= []; %str2double(systems_classified(j).Raindata(ind).Hour);
%          end
%              
%      end
    
    
end

 disp(' frequency calculations')   
toc; 
end



   

