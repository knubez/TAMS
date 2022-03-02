% Precipitation Asssignment
% DESCRPTION
% ----------------------
% This funtion utilizes IMERG V6 product to assign rainfall to each of the MCSs identified


% Addpath to external functions 
addpath('/gpfs/group/jle7/default/kmn18/graduateresearch/matlabcodes/cbrewer/cbrewer');
addpath('/gpfs/group/jle7/default/kmn18/graduateresearch/matlabcodes/borders');
addpath('/gpfs/group/jle7/default/kmn18/graduateresearch/matlabcodes/altmany-export_fig-2763b78'); % add export_fig

% Find data
clc; clear all; close all;
folder='/gpfs/group/jle7/default/kmn18/graduateresearch/NASA_IMERG_PRECIP_AUG_SEPT_2006/'; %3B-HHR.MS.MRG.3IMERG.*.nc4/'
Year='2006'; % systems(j).year;
Month='09'; % systems(j).month;
Day= '13';  %  systems(j).day;
Hour= '200000'; %
yearmonthday=strcat(Year,Month,Day);
list=dir([folder '3B-HHR.MS.MRG.3IMERG.' yearmonthday '-S' '083000' '*.nc4']);
filepath=[folder list.name];
ncid=netcdf.open(filepath);
datafile =list.name;
[~, filebasename, extension] = fileparts(datafile);
nombrecalc2era=filebasename;

varidp=netcdf.inqVarID(ncid,'precipitationCal'); % combined microwave-IR estimate with gauge calibration [mm /hr]
varidlat=netcdf.inqVarID(ncid,'lat'); % 353
varidlon=netcdf.inqVarID(ncid,'lon'); % 1175  [degrees east] -180 180

precip= double(netcdf.getVar(ncid,varidp));
%w=find(precip==0);
%precip(w)=NaN; 


latitude=double(netcdf.getVar(ncid,varidlat));
longitude=double(netcdf.getVar(ncid,varidlon));

%xlims= [-8 8];
%ylims=[ 12 17];

figure; % Plotting all the domain over test region 
load coastlines
coast = load('coast.mat');
borders('countries','nomap','k')
axis tight
[x,y]=meshgrid(longitude,latitude);
pcolor(x,y, precip);
shading interp
% restrict to box region
%xlim(xlims);
%ylim(ylims);



cmap=cbrewer('div','Spectral',64);
colormap(flipud(cmap));
%set(gca,'Ydir','reverse')
cmap(1:2,:)=1;

title('no interpolation all data')

% Steps for regridding and interp:

  % First define a meshgrid with the resolution desired ( EUMETSAT ~3KM (0.0degrees) over the large static doamin in study 
 
 delta = 0.035; % EUMETSAT 
 targetLats =ylims(1):delta:ylims(2); targetLons = xlims(1):delta:xlims(2); 

 [lonArray,latArray]=meshgrid(targetLons,targetLats);
 % Find large stattic doain in sutdy from precip data
   
 [n]=find(latitude>=ylims(1) & latitude<=ylims(2));
 [m]=find(longitude>=xlims(1) & longitude<=xlims(2));
 lat=latitude(n);
 lon=longitude(m);
 Precip=precip(n,m);
 [xx,yy]=meshgrid(lon,lat);
 
 figure; % plotting defined region no interpolation
load coastlines
coast = load('coast.mat');
borders('countries','nomap','k')
axis tight
pcolor(xx,yy,Precip);
shading interp
% restrict to box region
xlim(xlims);
ylim(ylims);
cmap=cbrewer('div','Spectral',64);
colormap(flipud(cmap));
colorbar
caxis([1 15])
title(' no interpolation' )
 
vq=interp2(lon,lat,Precip,lonArray,latArray,'linear');

figure; % plotting defined region linearly interpolated 
load coastlines
coast = load('coast.mat');
borders('countries','nomap','k')
axis tight
pcolor(lonArray,latArray,vq);
shading interp
% restrict to box region
xlim(xlims);
ylim(ylims);
cmap=cbrewer('div','Spectral',64);
colormap(flipud(cmap));
colorbar
caxis([1 15])
title(' linear interpolation' )

figure; % plotting defined region  nearest neighbor interpolated 
load coastlines
coast = load('coast.mat');
borders('countries','nomap','k')
axis tight
vq2=interp2(lon,lat,Precip,lonArray,latArray,'nearest');
pcolor(lonArray,latArray,vq2);
shading interp
% restrict to box region
xlim(xlims);
ylim(ylims);
cmap=cbrewer('div','Spectral',64);
colormap(flipud(cmap));
colorbar
caxis([1 15])
title(' nearest neighbor interpolation' )




 
                
