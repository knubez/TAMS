function [systems_classified ] = familyplots( systems_classified )
%familyplotsw-name of function
%----------------
% 1. It will take the indices of kids and parents of the original "newsystems" structure and convert them to the indices of the current new
%    structure of the systems called "systems_classified
% 2. This is done to be bale to plot each kid cloud element with its parent and be able to plot the lines correctly of the trajectories
%
% 3.
 
 
% HIST
% Created February  15th 2017 by Kelly M. Nunez Ocasio
% --------------------------------------------------------------------------------------------------------------------
% locations of files
addpath('/gpfs/group/jle7/default/kmn18/graduateresearch/matlabcodes/altmany-export_fig-2763b78'); % add export_fig
tic;
for k=1:length(systems_classified) % Loop over systems
    % if strcmp(systems_classified(k).Class,'DSL')==0
    % k
    county_splits=0;
    county_mergers=0;
    for l=1:length(systems_classified(k).System) % Loop over cloud elements of current system
        %    l
        
        if length(systems_classified(k).System(l).globalkids)>1
            county_splits=county_splits + 1;
        end
        
        for m=1:length(systems_classified(k).System(l).globalkids) %Loop over kids to find the indices of the global kid
            %       m
            systems_classified(k).System(l).localkids(m)=find(systems_classified(k).System(l).globalkids(m)==systems_classified(k).GlobalFamily);
        end
        
        if length(systems_classified(k).System(l).globalparent)>1
            county_mergers=county_mergers + 1;
        end
        for n=1:length(systems_classified(k).System(l).globalparent) %Loop over parents to find the indices of the global parent
            %      n
            %     disp(' curent global parent')
            % systems_classified(k).System(l).globalparent(n)
            %    disp('cureent global family')
            % systems_classified(k).GlobalFamily
            systems_classified(k).System(l).localparent(n)=find(systems_classified(k).System(l).globalparent(n)==systems_classified(k).GlobalFamily);
        end
    end
    % end
    systems_classified(k).splits= county_splits;
    systems_classified(k).mergers= county_mergers;
    
end
% h=figure;
% set(gcf,'color','w');
% %Large domain
%  lonlim = [-75 35]; 
%  latlim = [-5 45];
% 
% %      lonlim = [-8,8]; 
% %      latlim = [12,17];
% % lonlim = [-37,20];
% % latlim= [0,18];
% 
% axesm('mercator','MapLatLimit', latlim, 'MapLonLimit', lonlim+360, ...
%     'Grid','on','ParallelLabel','on','PLabelLocation',10,'MeridianLabel','on', ...
%     'MLabelParallel','south','MLabelLocation',15,'FontSize',18)
% hold on;
% coast = load('coast.mat');
% hold on;
% borders('countries','k')
% hold on;
% %plotm([12;12;17;17;12],[-8;8;8;-8;-8], 'r-', 'Linewidth', 2);
% hold on;
% plotm(coast.lat,coast.long,'-k')
% hold on;
% 
% %Family plot <3
% for k=1:length(systems_classified) % Loop over systems
%     if strcmp(systems_classified(k).Class,'DSL')==0
% 
%         colory=jet(length(systems_classified(k).System));
%         ry=1;
%         for l=1:length(systems_classified(k).System) % Loop over cloud elements of current system
%             current_lat=systems_classified(k).System(l).lat;
%             current_lon=systems_classified(k).System(l).lon;
%             
%             for m=1:length([systems_classified(k).System(l).localkids]) %Loop up current case's lat,lon
%                 current_kid_index=[systems_classified(k).System(l).localkids(m)];
%                 current_kid_lat=systems_classified(k).System(current_kid_index).lat;
%                 current_kid_lon=systems_classified(k).System(current_kid_index).lon;
%                 
%                 %
%                 %Loop over current case's kids
%                 
%                 % for n=1:length(systems_classified(k).System(l).localkids) %Loop over parents to find the indices of the global parent
%                 
%                 plotm([current_lat,current_kid_lat],[current_lon,...
%                     current_kid_lon],'^-k','LineWidth',1.5,'MarkerSize', 3);
%                 % hold on
%                 
%                 str_kid=num2str(current_kid_index);
%                 str=num2str(l);
%                 %textm(current_lat,current_lon,strcat({str}),'FontSize',15);
%                 
%             end
%             count=0;
%             if l>1
%                 %datetime here instead
%                 %current
%                 year =str2double(systems_classified(k).System(l).year );
%                 month  =str2double(systems_classified(k).System(l).month );
%                 day =str2double(systems_classified(k).System(l).day);
%                 hour =str2double(systems_classified(k).System(l).hour);
%                 minutes=str2double(systems_classified(k).System(l).minutes);
%                 seconds  =str2double(systems_classified(k).System(l).seconds);
%                 time=datetime(year,month,day,hour,minutes,seconds);
%                 %time_hour=hour(time);
%                 str=datestr(time);
%                 %         title(strcat({str},{' '},{'UTC'}),'Color',colors(timy,:),'FontSize',18);
%                 %         str=num2str(newsystems1(k).kids);
%                 %         str=num2str(newsystems1(k).kids(1));
%                % textm(systems_classified(k).System(l).lat+0.01,systems_classified(k).System(l).lon+0.20,strcat({str(13:20)},{' '},{'UTC'}),'FontSize',12);
%                 % previous
%                 year_before =str2double(systems_classified(k).System(l-1).year );
%                 month_before  =str2double(systems_classified(k).System(l-1).month );
%                 day_before =str2double(systems_classified(k).System(l-1).day);
%                 hour_before =str2double(systems_classified(k).System(l-1).hour);
%                 minutes_before=str2double(systems_classified(k).System(l-1).minutes);
%                 seconds_before  =str2double(systems_classified(k).System(l-1).seconds);
%                 time_before=datetime(year_before,month_before,day_before,hour_before,minutes_before,seconds_before);
%                 %time_hour_before=hour(time_before);
%                 str_before=datestr(time_before);
%                 
%                 
%                 if strcmp(str,str_before)==1
%                     count=count+1;
%                     
%                     hold on
%                     plotm(current_lat,current_lon,'^','MarkerFaceColor',colory(l-1,:),'MarkerSize', 3);
%                     str=num2str(l);
%                     % textm(current_lat,current_lon,strcat({str}),'FontSize',15);
%                 else
%                     ry=ry+1;
%                     
%                     hold on
%                     plotm(current_lat,current_lon,'^','MarkerFaceColor',colory(l,:),'MarkerSize', 3);
%                     str=num2str(l);
%                     % textm(current_lat,current_lon,strcat({str}),'FontSize',15);
%                 end
%             else
%                 year =str2double(systems_classified(k).System(l).year );
%                 month  =str2double(systems_classified(k).System(l).month );
%                 day =str2double(systems_classified(k).System(l).day);
%                 hour =str2double(systems_classified(k).System(l).hour);
%                 minutes=str2double(systems_classified(k).System(l).minutes);
%                 seconds  =str2double(systems_classified(k).System(l).seconds);
%                 time=datetime(year,month,day,hour,minutes,seconds);
%                 %time_hour=hour(time);
%                 str=datestr(time);
%                 %         title(strcat({str},{' '},{'UTC'}),'Color',colors(timy,:),'FontSize',18);
%                 %         str=num2str(newsystems1(k).kids);
%                 %         str=num2str(newsystems1(k).kids(1));
%                 %textm(systems_classified(k).System(l).lat+0.01,systems_classified(k).System(l).lon+0.20,strcat({str(13:20)},{' '},{'UTC'}),'FontSize',12);
%                 %
%                 hold on
%                 plotm(current_lat,current_lon,'^','MarkerFaceColor',colory(1,:),'MarkerSize', 3);
%                 str=num2str(l);
%                 % textm(current_lat,current_lon,strcat({str}),'FontSize',15);
%                 ry=ry+1;
%             end
%             
%         end
%          title('August -September, 2006 MCSs','FontSize',18)
%         
%         %     frame = getframe(h);
%         %     %county=county+1;
%         %     im = frame2im(frame);
%         %     %figname=['/gpfs/group/jle7/default/kmn18/graduateresearch/outputfiles/points235nopro_',num2str(namefile(1:8)),'_',num2str(county)];
%         %     saveas(h,figname, 'png')
%         %     %     alpha=0.4;
%         %     %     set(im, 'AlphaData', alpha);
%         %     [imind,cm] = rgb2ind(im,256);
%         %     if  k==1
%         %         imwrite(imind,cm,filename,'gif','Loopcount',inf);
%         %     else
%         %         imwrite(imind,cm,filename,'gif','WriteMode','append');
%         %     end
%         %     hold
%        % hold off;
%     end
% end
%Save figure
% export_fig(h,'/gpfs/group/jle7/default/kmn18/graduateresearch/figures/figuresAug_Sept_06_largedomain/familyplot', '-r300');
disp('Family connections done')
toc;
end