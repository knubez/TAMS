% Only runs if systems_classified is loaded
addpath('/gpfs/group/jle7/default/kmn18/graduateresearch/matlabcodes/TAMS_precip_IMERGV6/');

% set(gcf,'color','w');
%Large domain
lonlim = [20 32];
latlim = [-8 20];

% DLL k=7
%  xlims =[-33 -28];
%  ylims=[4.5, 8.5];

%CCC k=4
xlims =[21 33];
ylims=[9, 18];

% xlims =[-35 45];
% ylims=[0, 30]

xlimss =[-35 45];
ylimss=[0, 30];

k=4;
rr=figure('Position',[660 -4 583 959]);
[ha pos]=tight_subplot(4,1,[.01 .03],[.1 .01],[.01 .01]);

d=unique({systems_classified(k).System.hour});
for l=1:length(d)-1 % Loop over cloud elements of current system
    idx=find(strcmp({systems_classified(k).System.hour},d(l))==1);
    
    if idx==1 % if its the first time plot the first 
          

        axes(ha(1))
        load coastlines
        coast = load('coast.mat');
        borders('countries','nomap','k')
        axis tight
        xlim(xlims)
        ylim(ylims)
        set(gca,'FontSize',13)
        set(gcf,'color','w')
        hold on
                
        % BT
        % Find satelite image of brightness temperature same date and time %
        databasedir_sat ='/gpfs/group/jle7/default/kmn18/graduateresearch/MSG_2006/MSG_IR_Aug_Sept_2006/archive.eumetsat.int/umarf/onlinedownload/KellyNunez/1268394/1/';
        files_sat=dir([databasedir_sat, 'W_XX-EUMETSAT-Darmstadt,VIS+IR+IMAGERY,MSG*+SEVIRI_C_EUMG_',...
            systems_classified(k).System(idx).year,systems_classified(k).System(idx).month,systems_classified(k).System(idx).day,systems_classified(k).System(idx).hour,'*.nc' ]);
        % Set name of current file to a variable ex. 3B42.2...
        datafile = files_sat.name;
        [~,basename, extension] = fileparts(datafile);
        
        %Adding the data file to the data path % concatenate folder name with specific data name
        filetoload = strcat(databasedir_sat, datafile);
        
        % Open netCDF file and load the things
        ncid = netcdf.open(filetoload);
        
        % % Get information about the contents of the file
        [ndims,nvars,ngatts,unlimdimid] = netcdf.inq(ncid); %inqVar(ncid)-returns information about variable %ndims-number of dimensions %
        
        
        % Get values for each variable SEVIRI channel 10.8 which is channel 9
        Lat = netcdf.getVar(ncid,140); %degress
        Lon = netcdf.getVar(ncid,141); %degrees
        ch9 = netcdf.getVar(ncid,137); % counts
        
        % convert to double precision
        Lat = double(Lat);
        Lon = double(Lon);
        
        % Attributes associated with variable Temp
        scale_factor = ncreadatt(filetoload,'ch9','scale_factor');
        add_offset = ncreadatt(filetoload,'ch9','add_offset');
        
        % Converting counts to radiance
        R= add_offset + (ch9 * scale_factor);
        
        % Converting from Radiance to Brightness Temp
        Temp=(( ( 1.43877 ) * ( 930.659 ) )./log ( ( ( (1.19104*10^(-5) ) * ( (930.659)^3 ) )./double(R) ) + 1 )-0.627)./0.9983;
        TempB=real(Temp);
        
        %PLot
        %f1=figure;
        Lon(Lon== -999)=NaN;
        Lat(Lat== -999)=NaN;
        
        pcolor(Lon,Lat,TempB);
        colormap(jet)
        hold on
        shading interp
        alpha(0.75)

        %hh=colorbar;
        caxis([190 300])
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
       % set(gca,'ytick',[])
     
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
    %set(gca,'ytick',[])
    
    
    for i=1:length(idx)
        current_lat=systems_classified(k).System(idx(i)).lat;
        current_lon=systems_classified(k).System(idx(i)).lon;       
        current_kid_index=[systems_classified(k).System(idx(i)).localkids];
        hold on
             
             if i==1
               
                %houry=num2str(houry);
                
                houry=str2double(systems_classified(k).System(idx(1)).hour)+2;
                if houry>9 
                
                databasedir_sat ='/gpfs/group/jle7/default/kmn18/graduateresearch/MSG_2006/MSG_IR_Aug_Sept_2006/archive.eumetsat.int/umarf/onlinedownload/KellyNunez/1268394/1/';
                files_sat=dir([databasedir_sat, 'W_XX-EUMETSAT-Darmstadt,VIS+IR+IMAGERY,MSG*+SEVIRI_C_EUMG_',...
                    systems_classified(k).System(idx(1)).year,systems_classified(k).System(idx(1)).month,systems_classified(k).System(idx(1)).day,num2str(houry),'*.nc' ]);
                else
                       databasedir_sat ='/gpfs/group/jle7/default/kmn18/graduateresearch/MSG_2006/MSG_IR_Aug_Sept_2006/archive.eumetsat.int/umarf/onlinedownload/KellyNunez/1268394/1/';
                files_sat=dir([databasedir_sat, 'W_XX-EUMETSAT-Darmstadt,VIS+IR+IMAGERY,MSG*+SEVIRI_C_EUMG_',...
                    systems_classified(k).System(idx(1)).year,systems_classified(k).System(idx(1)).month,systems_classified(k).System(idx(1)).day,'0',num2str(houry),'*.nc' ]); 
                end
            
                % Set name of current file to a variable ex. 3B42.2...
                datafile = files_sat.name;
                [~,basename, extension] = fileparts(datafile);
                
                disp('name of file')
                basename
                %Adding the data file to the data path % concatenate folder name with specific data name
                filetoload = strcat(databasedir_sat, datafile);
                
                % Open netCDF file and load the things
                ncid = netcdf.open(filetoload);
                
                % % Get information about the contents of the file
                [ndims,nvars,ngatts,unlimdimid] = netcdf.inq(ncid); %inqVar(ncid)-returns information about variable %ndims-number of dimensions %
                
                
                % Get values for each variable SEVIRI channel 10.8 which is channel 9
                Lat = netcdf.getVar(ncid,140); %degress
                Lon = netcdf.getVar(ncid,141); %degrees
                ch9 = netcdf.getVar(ncid,137); % counts
                
                % convert to double precision
                Lat = double(Lat);
                Lon = double(Lon);
                
                % Attributes associated with variable Temp
                scale_factor = ncreadatt(filetoload,'ch9','scale_factor');
                add_offset = ncreadatt(filetoload,'ch9','add_offset');
                
                % Converting counts to radiance
                R= add_offset + (ch9 * scale_factor);
                
                % Converting from Radiance to Brightness Temp
                Temp=(( ( 1.43877 ) * ( 930.659 ) )./log ( ( ( (1.19104*10^(-5) ) * ( (930.659)^3 ) )./double(R) ) + 1 )-0.627)./0.9983;
                TempB=real(Temp);
                
                %PLot
                %f1=figure;
                Lon(Lon== -999)=NaN;
                Lat(Lat== -999)=NaN;
                
                pcolor(Lon,Lat,TempB);
                colormap(jet)
                
                
                caxis([190 300])
                shading interp
                alpha(0.75)
             end
        

        
        hold on
        for m=1:length(current_kid_index)
            current_kid_lat=systems_classified(k).System(current_kid_index(m)).lat;
            current_kid_lon=systems_classified(k).System(current_kid_index(m)).lon;

            hold on
            
            plot([current_lon,current_kid_lon],[current_lat,current_kid_lat],'^-k','LineWidth',1.5,'MarkerSize', 4);
            
            hold on
            plot(systems_classified(k).System(current_kid_index(m)).x, systems_classified(k).System(current_kid_index(m)).y,'-k','LineWidth',1.5);
            
            hold on
            text(current_kid_lon-0.05,current_kid_lat-0.4,{systems_classified(k).System(current_kid_index(m)).hour},'FontSize',12);
            
        
        end
        hold on
        plot(systems_classified(k).System(idx(i)).lon, systems_classified(k).System(idx(i)).lat,'^k','LineWidth',1.5,'MarkerSize', 4);
        hold on
        text(current_lon-0.05,current_lat-0.4,{systems_classified(k).System(idx(i)).hour},'FontSize',12);
        hold on

        set(gca,'xtick',[])
        %set(gca,'ytick',[])
        set(gca,'Box','on');
    end
  
end
h=colorbar;
%title(h,'[K]');
h.Location='southoutside';
%h.Position=[0.1304 0.0553 0.7753 0.0127];
h.Position=[0.1304 0.0353 0.7753 0.0127];
set(ha(length(ha)),'XTickMode','auto','XTickLabelMode','auto')
xlabel ('Longitude [degrees east]')
%set(ha,'YTickMode','auto','YTickLabelMode','auto')
xlabel ('Longitude [degrees east]')