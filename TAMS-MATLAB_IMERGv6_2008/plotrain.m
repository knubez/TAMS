% Only runs if systems_classified is loaded
addpath('/gpfs/group/jle7/default/kmn18/graduateresearch/matlabcodes/TAMS_precip_IMERGV6/');

% set(gcf,'color','w');
%Large domain
lonlim = [20 32];
latlim = [-8 20];

% CCC
xlims =[21 33];
ylims=[9, 18];

% DLL
%  xlims =[-33 -28];
%  ylims=[4.5, 8.5];

xlimss =[-35 45];
ylimss=[0, 30];

% % 
% figure; % Plotting all the domain over test region
% load coastlines
% coast = load('coast.mat');
% borders('countries','nomap','k')
% axis tight
% xlim(xlims)
% ylim(ylims)
% 
% title('CCC identified  on 2006-09-11')
% xlabel('Longitude')
% ylabel('Latitude')
% set(gca,'FontSize',20)
% set(gcf,'color','w')

k=4;
rr=figure('Position',[660 -4 583 959]);

%xlabel('Longitude')
%('Latitude')
[ha pos]=tight_subplot(4,1,[.01 .03],[.1 .01],[.01 .01]);
d=unique({systems_classified(k).System.hour});
for l=1:length(d)-1 % Loop over cloud elements of current system
    idx=find(strcmp({systems_classified(k).System.hour},d(l))==1);
    
    if idx==1 % if its the first time plot the first 
          
        %figure;% Plotting all the domain over test region
%         subplot(length(d),1,l)
        axes(ha(1))
        load coastlines
        coast = load('coast.mat');
        borders('countries','nomap','k')
        axis tight
        xlim(xlims)
        ylim(ylims)
        
        %title({'CCC identified  on 2006-09-11'}); %,systems_classified(k).System(idx).hour})
        %xlabel('Longitude')
        %ylabel('Latitude')
        set(gca,'FontSize',13)
        set(gcf,'color','w')
        hold on
        % First define a meshgrid with the resolution desired ( EUMETSAT ~3KM (0.0degrees) over the large static doamin in study
        delta = 0.035; % EUMETSAT ~ 3km
        targetLats =ylimss(1):delta:ylimss(2); targetLons = xlimss(1):delta:xlimss(2);
        [lonArray,latArray]=meshgrid(targetLons,targetLats);
        new_lon = lonArray.*systems_classified(k).Raindata(idx).indices_in;
        new_lat = latArray.*systems_classified(k).Raindata(idx).indices_in ;
        pcolor(new_lon,new_lat,[systems_classified(k).Raindata(idx).rain_rate_2])
        shading interp
        %alpha(0.5)
        cmap=cbrewer('div','Spectral',64);
        colormap(flipud(cmap));
        %hh=colorbar;
        caxis([5 16])
       % title(hh,'[mm hr^{-1}]');
       
        current_lat=systems_classified(k).System(idx).lat;
        current_lon=systems_classified(k).System(idx).lon;
        hold on
        plot(systems_classified(k).System(idx).x, systems_classified(k).System(idx).y,'-k','LineWidth',1.5);
        
        hold on
        
        plot(systems_classified(k).System(idx).lon, systems_classified(k).System(idx).lat,'^k','LineWidth',1.5,'MarkerSize', 4);
        hold on
        text(current_lon-0.05,current_lat-0.4,{systems_classified(k).System(idx).hour},'FontSize',12)
        set(gca,'xtick',[])
        set(gca,'ytick',[])
     
        set(gca,'Box','on');
    end
    
    %figure; % Plotting all the domain over test region
    %subplot(length(d),1,l+1)
    axes(ha(l+1))
    load coastlines
    coast = load('coast.mat');
    borders('countries','nomap','k')
    axis tight
    xlim(xlims)
    ylim(ylims)
    
    t=systems_classified(k).System(idx(1)).hour;
    t=str2double(t);
    t=t+2;
    %title({'CCC identified  on 2006-09-11',num2str(t), ':00 UTC'})
   % xlabel('Longitude')
    %ylabel('Latitude')
    set(gca,'FontSize',13)
    set(gcf,'color','w')
    hold on
    %grid on
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    for i=1:length(idx)
        current_lat=systems_classified(k).System(idx(i)).lat;
        current_lon=systems_classified(k).System(idx(i)).lon;       
        current_kid_index=[systems_classified(k).System(idx(i)).localkids];
        for m=1:length(current_kid_index)
            current_kid_lat=systems_classified(k).System(current_kid_index(m)).lat;
            current_kid_lon=systems_classified(k).System(current_kid_index(m)).lon;
            
            hold on
            % First define a meshgrid with the resolution desired ( EUMETSAT ~3KM (0.0degrees) over the large static doamin in study
            delta = 0.035; % EUMETSAT ~ 3km
            targetLats =ylimss(1):delta:ylimss(2); targetLons = xlimss(1):delta:xlimss(2);
            [lonArray,latArray]=meshgrid(targetLons,targetLats);
            new_lon = lonArray.*systems_classified(k).Raindata(current_kid_index(m)).indices_in;
            new_lat = latArray.*systems_classified(k).Raindata(current_kid_index(m)).indices_in ;
            pcolor(new_lon,new_lat,[systems_classified(k).Raindata(current_kid_index(m)).rain_rate_2])
            shading interp
            %alpha(0.5)
            cmap=cbrewer('div','Spectral',64);
            colormap(flipud(cmap));
            %hh=colorbar;
            caxis([5 16])
            %
            
            hold on
            
            plot([current_lon,current_kid_lon],[current_lat,current_kid_lat],'^-k','LineWidth',1.5,'MarkerSize', 4);
            
            hold on
            plot(systems_classified(k).System(current_kid_index(m)).x, systems_classified(k).System(current_kid_index(m)).y,'-k','LineWidth',1.5);
            
            hold on
            text(current_kid_lon-0.05,current_kid_lat-0.4,{systems_classified(k).System(current_kid_index(m)).hour},'FontSize',12);
            
        
        end
        plot(systems_classified(k).System(idx(i)).lon, systems_classified(k).System(idx(i)).lat,'^k','LineWidth',1.5,'MarkerSize', 4);
        hold on
        text(current_lon-0.05,current_lat-0.4,{systems_classified(k).System(idx(i)).hour},'FontSize',12);
        hold on
      %  grid on
       
      %set(gca,'xtick',[])
      %set(gca,'ytick',[])
      
        set(gca,'Box','on');
    end
    h=colorbar;
    c=title(h,'[mm hr^{-1}]');
    c.FontSize=10;
    set(c,'position',[5 13 0])
    h.Location='southoutside';
    %h.Position=[0.1304 0.0453 0.7753 0.0127];
    h.Position=[0.1304 0.0353 0.7753 0.0127];
    set(ha(length(ha)),'XTickMode','auto','XTickLabelMode','auto')
    xlabel ('Longitude [degrees east]')
    %ha=get(gcf,'children');
%     set(ha(1),'position',[.5 .1 .4 .4])
%     set(ha(2),'position',[.1 .1 .4 .4])
%     set(ha(2),'position',[.5 .1 .4 .4])
%     set(ha(2),'position',[.1 .5 .4 .4])
end
